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
import logging
import tempfile
from pathlib import Path
from typing import Optional, Type

import jsonlines

from model_navigator.api.config import ProfilerConfig
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.performance.performance import Profiler, ProfilingResults
from model_navigator.execution_context import ExecutionContext
from model_navigator.logger import LOGGER
from model_navigator.runners.base import NavigatorRunner
from model_navigator.utils.common import parse_kwargs_to_cmd
from model_navigator.utils.tensor import TensorMetadata


class MaxBatchSizeFinder(Profiler):
    """Overriden profiled for max batch size search."""

    @property
    def _profiling_results_logging_level(self):
        return logging.DEBUG


class FindMaxBatchSize(Command):
    """Command for searching maximal possible batch size that model can be loaded with."""

    def _run(
        self,
        workspace: Path,
        path: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        verbose: bool,
        runner_cls: Type[NavigatorRunner],
        reproduce_script_dir: Optional[Path] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where artifacts are stored
            path: Path to the model for execution
            input_metadata: Information about model inputs
            output_metadata: Information about model outputs
            batch_dim: Place where batch dimension is located in shape
            verbose: Flag to enable/disable verbose logging for command
            runner_cls: Runner used for model execution
            reproduce_script_dir: Script where reproduction of command is saved

        Returns:
            CommandOutput with status and additional output parameters
        """
        model_path = workspace / path
        model_dir = model_path.parent
        reproduce_script_dir = reproduce_script_dir or model_dir

        if not model_path.exists():
            LOGGER.warning(f"Model: {model_path.as_posix()!r} not found, command skipped.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        if batch_dim is None:
            LOGGER.info("Model does not support batching.")
            return CommandOutput(status=CommandStatus.OK, output={"device_max_batch_size": None})

        profiler_config = ProfilerConfig(
            max_trials=1,
            measurement_request_count=1,
            throughput_cutoff_threshold=-2,
        )

        with ExecutionContext(
            workspace=workspace,
            script_path=reproduce_script_dir / "reproduce_max_batch_size.py",
            cmd_path=reproduce_script_dir / "reproduce_max_batch_size.sh",
            verbose=verbose,
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "navigator_workspace": workspace.as_posix(),
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "profiler_config": profiler_config.to_dict(parse=True),
                "model_path": path,
                "runner_name": runner_cls.name(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
            }

            args = parse_kwargs_to_cmd(kwargs)

            from model_navigator.commands.find_max_batch_size import find_max_batch_size_script

            try:
                context.execute_external_runtime_script(find_max_batch_size_script.__file__, args)
            except Exception:
                pass

            with jsonlines.open(temp_file.name, "r") as f:
                results = [ProfilingResults.from_dict(res) for res in f]

            if not results:
                LOGGER.info("Max batch size not found. Please review the log.")
                device_max_batch_size = None
            else:
                device_max_batch_size = results[-1].batch_size
                LOGGER.info(f"Device max batch size: {device_max_batch_size}")

        return CommandOutput(status=CommandStatus.OK, output={"device_max_batch_size": device_max_batch_size})
