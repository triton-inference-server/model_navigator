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
"""Command for performance measurement."""

import pathlib
import shutil
import tempfile
from typing import Any, Optional, Type

from jsonlines import jsonlines

from model_navigator.api.config import Format, OptimizationProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.performance.results import ProfilingResults
from model_navigator.configuration.runner.runner_config import RunnerConfig
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorProfilingError
from model_navigator.runners.base import NavigatorRunner
from model_navigator.utils.common import parse_kwargs_to_cmd
from model_navigator.utils.format_helpers import is_source_format


class Performance(Command):
    """Performance command."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        format: Format,
        optimization_profile: OptimizationProfile,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        verbose: bool,
        runner_cls: Type[NavigatorRunner],
        reproduce_script_dir: Optional[pathlib.Path] = None,
        model: Optional[Any] = None,
        runner_config: Optional[RunnerConfig] = None,
    ) -> CommandOutput:
        """Run performance command.

        Args:
            workspace: Model Navigator workspace path.
            path: Model path, relative to the workspace.
            format: Model format.
            optimization_profile: Optimization profile used during conversion and profiling.
            input_metadata: Input metadata.
            output_metadata: Output metadata.
            batch_dim: Batch dimension.
            verbose: If True verbose logging.
            runner_cls: Runner type to profile the model with.
            reproduce_script_dir: Path to store the reproducing scripts for the command.
                When None use model directory. Defaults to None.
            model: Model when profiling on a source format. Defaults to None.
            runner_config: Additional runner arguments.

        Returns:
            CommandOutput: Output of the command containing profiling results.
        """
        model_path = workspace.path / path
        model_dir = model_path.parent
        reproduce_script_dir = reproduce_script_dir or model_dir

        if not is_source_format(format) and not model_path.exists():
            LOGGER.warning(f"Model: {model_path.as_posix()!r} not found, command skipped.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        profiler_samples = workspace.path / "model_input" / "profiler"
        if profiler_samples.exists():
            shutil.rmtree(profiler_samples.as_posix())

        profiling_samples = workspace.path / "model_input" / "profiling"
        shutil.copytree(profiling_samples, profiler_samples)

        with ExecutionContext(
            workspace=workspace,
            script_path=reproduce_script_dir / f"reproduce_profiling-{runner_cls.slug()}.py",
            cmd_path=reproduce_script_dir / f"reproduce_profiling-{runner_cls.slug()}.sh",
            verbose=verbose,
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "navigator_workspace": workspace.path.as_posix(),
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "runner_name": runner_cls.name(),
                "optimization_profile": optimization_profile.to_dict(parse=True),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
                "runner_config": runner_config.to_dict(parse=True) if runner_config else None,
            }

            from model_navigator.commands.performance import profile_script

            if is_source_format(format):
                profile_script.get_model = lambda: model
                args = parse_kwargs_to_cmd(kwargs)
                context.execute_local_runtime_script(
                    profile_script.__file__, profile_script.profile, args, allow_failure=True
                )
            else:
                kwargs["model_path"] = path
                args = parse_kwargs_to_cmd(kwargs)
                context.execute_external_runtime_script(profile_script.__file__, args, allow_failure=True)
            with jsonlines.open(temp_file.name, "r") as f:
                profiling_results = [ProfilingResults.from_dict(res) for res in f]
        if not profiling_results:
            raise ModelNavigatorProfilingError("No profiling results found.")
        return CommandOutput(status=CommandStatus.OK, output={"profiling_results": profiling_results})
