# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality to perform search of max possible batch size that model can be loaded with on device."""

import dataclasses
import logging
import pathlib
import tempfile
from typing import Dict, List, Optional, Type, Union

import jsonlines

from model_navigator.api.config import OptimizationProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.performance import Profiler, ProfilingResults
from model_navigator.configuration.runner.runner_config import RunnerConfig
from model_navigator.core.constants import DEFAULT_MAX_BATCH_SIZE_THRESHOLD
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.runners.base import NavigatorRunner
from model_navigator.utils.common import parse_kwargs_to_cmd


class MaxBatchSizeFinder(Profiler):
    """Overridden profiled for max batch size search."""

    @property
    def _profiling_results_logging_level(self):
        return logging.DEBUG


@dataclasses.dataclass
class FindMaxBatchSizeConfig:
    """Configure pairs of model config and runner to execute."""

    model_path: Union[str, pathlib.Path]
    runner_cls: Type[NavigatorRunner]


class FindMaxBatchSize(Command):
    """Command for searching maximal possible batch size that model can be loaded with."""

    def _run(
        self,
        configurations: List[FindMaxBatchSizeConfig],
        workspace: Workspace,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        optimization_profile: OptimizationProfile,
        verbose: bool,
        reproduce_script_dir: Optional[pathlib.Path] = None,
        runner_config: Optional[Dict] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            configurations: Configuration to use during search
            workspace: Workspace where artifacts are stored
            input_metadata: Information about model inputs
            output_metadata: Information about model outputs
            batch_dim: Place where batch dimension is located in shape
            optimization_profile: Configuration for performance measurement
            verbose: Flag to enable/disable verbose logging for command
            reproduce_script_dir: Script where reproduction of command is saved
            runner_config: Additional runner configuration.

        Returns:
            CommandOutput with status and additional output parameters
        """
        device_max_batch_size = None
        for configuration in configurations:
            device_max_batch_size = self._execute_configuration(
                workspace=workspace,
                path=configuration.model_path,
                input_metadata=input_metadata,
                output_metadata=output_metadata,
                batch_dim=batch_dim,
                optimization_profile=optimization_profile,
                verbose=verbose,
                runner_cls=configuration.runner_cls,
                reproduce_script_dir=reproduce_script_dir,
                runner_config=runner_config,
            )
            if device_max_batch_size is not None:
                break

        return CommandOutput(
            status=CommandStatus.OK,
            output={"device_max_batch_size": device_max_batch_size},
        )

    def _execute_configuration(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        optimization_profile: OptimizationProfile,
        verbose: bool,
        runner_cls: Type[NavigatorRunner],
        reproduce_script_dir: Optional[pathlib.Path] = None,
        runner_config: Optional[RunnerConfig] = None,
    ):
        model_path = workspace.path / path
        model_dir = model_path.parent
        reproduce_script_dir = reproduce_script_dir or model_dir
        device_max_batch_size = None

        if not model_path.exists():
            LOGGER.warning(f"Model: {model_path.as_posix()!r} not found, command skipped.")
            return device_max_batch_size

        if batch_dim is None:
            LOGGER.info("Model does not support batching.")
            return device_max_batch_size

        if optimization_profile.max_batch_size or optimization_profile.batch_sizes:
            if optimization_profile.max_batch_size:
                device_max_batch_size = optimization_profile.max_batch_size
            else:
                device_max_batch_size = max(optimization_profile.batch_sizes)
            LOGGER.info(f"Using maximal batch size provided in optimization profile: {device_max_batch_size}")
            return device_max_batch_size

        optimization_profile = OptimizationProfile(
            min_trials=1,
            max_trials=1,
            window_size=1,
            stabilization_windows=1,
            throughput_cutoff_threshold=-2,
        )

        with ExecutionContext(
            workspace=workspace,
            script_path=reproduce_script_dir / f"reproduce_max_batch_size-{runner_cls.slug()}.py",
            cmd_path=reproduce_script_dir / f"reproduce_max_batch_size-{runner_cls.slug()}.sh",
            verbose=verbose,
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "navigator_workspace": workspace.path.as_posix(),
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "optimization_profile": optimization_profile.to_dict(parse=True),
                "model_path": path,
                "runner_name": runner_cls.name(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
                "runner_config": runner_config.to_dict(parse=True) if runner_config else None,
            }

            args = parse_kwargs_to_cmd(kwargs)

            from model_navigator.commands.find_max_batch_size import find_max_batch_size_script

            try:
                context.execute_external_runtime_script(find_max_batch_size_script.__file__, args, allow_failure=True)
            except Exception:
                pass

            with jsonlines.open(temp_file.name, "r") as f:
                results = [ProfilingResults.from_dict(res) for res in f]

            if not results:
                LOGGER.info("Max batch size not found. Please review the log.")
                device_max_batch_size = None
            else:
                device_max_batch_size = results[-1].batch_size
                if device_max_batch_size > DEFAULT_MAX_BATCH_SIZE_THRESHOLD:
                    device_max_batch_size = device_max_batch_size // 2

                LOGGER.info(f"Found device max batch size: {device_max_batch_size}.")

            return device_max_batch_size
