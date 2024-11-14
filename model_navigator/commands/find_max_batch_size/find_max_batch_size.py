# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Type, Union

import jsonlines

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.performance import ProfilingResults
from model_navigator.configuration import DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD, Format, OptimizationProfile
from model_navigator.configuration.runner.runner_config import RunnerConfig
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorRuntimeError
from model_navigator.runners.base import NavigatorRunner
from model_navigator.utils.common import parse_kwargs_to_cmd
from model_navigator.utils.format_helpers import is_source_format


@dataclasses.dataclass
class FindMaxBatchSizeConfig:
    """Configure pairs of model config and runner to execute."""

    format: Format
    runner_cls: Type[NavigatorRunner]
    reproduction_scripts_dir: pathlib.Path
    model: Any = None
    model_path: Union[str, pathlib.Path, None] = None


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
            runner_config: Additional runner configuration.

        Returns:
            CommandOutput with status and additional output parameters
        """
        device_max_batch_size = None
        if batch_dim is None:
            LOGGER.info("Model does not support batching.")
            return CommandOutput(
                status=CommandStatus.OK,
                output={"device_max_batch_size": device_max_batch_size},
            )

        if optimization_profile.max_batch_size or optimization_profile.batch_sizes:
            if optimization_profile.max_batch_size:
                device_max_batch_size = optimization_profile.max_batch_size
            else:
                device_max_batch_size = max(optimization_profile.batch_sizes)
            LOGGER.info(f"Using maximal batch size provided in optimization profile: {device_max_batch_size}")
            return CommandOutput(
                status=CommandStatus.OK,
                output={"device_max_batch_size": device_max_batch_size},
            )

        device_max_batch_sizes = []
        for configuration in configurations:
            executor_max_batch_size = self._execute_configuration(
                workspace=workspace,
                format=configuration.format,
                model=configuration.model,
                path=configuration.model_path,
                reproduction_scripts_dir=configuration.reproduction_scripts_dir,
                input_metadata=input_metadata,
                output_metadata=output_metadata,
                batch_dim=batch_dim,
                optimization_profile=optimization_profile,
                verbose=verbose,
                runner_cls=configuration.runner_cls,
                runner_config=runner_config,
            )
            if executor_max_batch_size is not None:
                device_max_batch_sizes.append(executor_max_batch_size)

        device_max_batch_size = max(device_max_batch_sizes) if device_max_batch_sizes else None

        if device_max_batch_size is None:
            LOGGER.warning(
                """Unable to find max batch size for TensorRT conversion based on base formats."""
                """Please provide max batch size in configuration to run TensorRT conversion with batch size > 1."""
            )
        else:
            LOGGER.info(f"""Max batch size for TensorRT conversion: {device_max_batch_size}.""")
            LOGGER.info(f"""Max batch size for performance profiling: {device_max_batch_size}.""")
            optimization_profile.max_batch_size = device_max_batch_size

        return CommandOutput(
            status=CommandStatus.OK,
            output={"device_max_batch_size": device_max_batch_size},
        )

    def _execute_configuration(
        self,
        workspace: Workspace,
        format: Format,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        optimization_profile: OptimizationProfile,
        verbose: bool,
        runner_cls: Type[NavigatorRunner],
        reproduction_scripts_dir: pathlib.Path,
        model: Any = None,
        path: Optional[pathlib.Path] = None,
        runner_config: Optional[RunnerConfig] = None,
    ):
        if not model and not path:
            raise ModelNavigatorRuntimeError("`mode` or `path` must be provided")

        if model is not None and path is not None:
            raise ModelNavigatorRuntimeError("Only one of `model` and `path` argument can be provided.")

        if not is_source_format(format) and path is None:
            raise ModelNavigatorRuntimeError("For serializable formats `path` argument is required.")

        if is_source_format(format) and model is None:
            raise ModelNavigatorRuntimeError("For source formats `model` argument is required.")

        if not is_source_format(format):
            model_path = workspace.path / path
        else:
            model_path = None

        device_max_batch_size = None
        if not is_source_format(format) and model_path and not model_path.exists():
            LOGGER.warning(f"Model: {model_path.as_posix()!r} not found, command skipped.")
            return device_max_batch_size

        LOGGER.info("Executing heuristic search based on existing model formats.")
        if model is not None:
            LOGGER.info(f"Selected framework model with {runner_cls}.")
        else:
            LOGGER.info(f"Selected {format.value} model in {path} with {runner_cls}.")

        reproduction_scripts_dir = workspace.path / reproduction_scripts_dir
        reproduction_scripts_dir.mkdir(exist_ok=True)

        optimization_profile_copy = optimization_profile.clone()
        if optimization_profile_copy.throughput_cutoff_threshold is None:
            optimization_profile_copy.throughput_cutoff_threshold = DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD
            LOGGER.info(
                f"""Using default throughput_cutoff_threshold={DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD} """
                """for heuristic search as `None` was provided."""
            )

        with ExecutionContext(
            workspace=workspace,
            script_path=reproduction_scripts_dir / f"reproduce_max_batch_size-{runner_cls.slug()}.py",
            cmd_path=reproduction_scripts_dir / f"reproduce_max_batch_size-{runner_cls.slug()}.sh",
            verbose=verbose,
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "navigator_workspace": workspace.path.as_posix(),
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "optimization_profile": optimization_profile_copy.to_dict(parse=True),
                "runner_name": runner_cls.name(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
                "runner_config": runner_config.to_dict(parse=True) if runner_config else None,
            }

            from model_navigator.commands.find_max_batch_size import find_max_batch_size_script

            if is_source_format(format):
                find_max_batch_size_script.get_model = lambda: model
                run_in_isolation = False
            elif model_path is not None:
                kwargs["model_path"] = model_path.as_posix()
                run_in_isolation = True
            else:
                raise ModelNavigatorRuntimeError("Incorrect configuration for model runner.")

            try:
                context.execute_python_script(
                    find_max_batch_size_script.__file__,
                    find_max_batch_size_script.find_max_batch_size,
                    args=parse_kwargs_to_cmd(kwargs),
                    allow_failure=True,
                    run_in_isolation=run_in_isolation,
                )
            except Exception:
                pass

            with jsonlines.open(temp_file.name, "r") as f:
                results = [ProfilingResults.from_dict(res) for res in f]

            if not results:
                LOGGER.info("Max batch size not found. Please review the log.")
                device_max_batch_size = None
            else:
                batch_sizes = [result.batch_size for result in results]
                device_max_batch_size = batch_sizes[-1]
                LOGGER.info(f"Found device max batch size: {batch_sizes}. Selected: {device_max_batch_size}")

            return device_max_batch_size
