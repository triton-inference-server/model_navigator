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
"""Runtime verification command."""


import pathlib
from typing import Any, Optional, Type

from model_navigator.api.config import Format, SizedDataLoader, VerifyFunction
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.correctness import Correctness
from model_navigator.commands.performance import Performance
from model_navigator.core.dataloader import extract_sample
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.frameworks import Framework
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import get_runner
from model_navigator.runners.utils import get_source_default_runners
from model_navigator.utils.format_helpers import FRAMEWORK2BASE_FORMAT


class VerifyModel(Command, requires=[Correctness.name, Performance.name]):
    """Verify model with a runner."""

    def _pre_run(
        self,
        verify_func: VerifyFunction,
    ) -> bool:
        """Check if command should be run.

        Args:
            verify_func (VerifyFunction): verification function.

        Returns:
            bool: True if command should be run, False otherwise.
        """
        if verify_func is None:
            LOGGER.info("verify_function not provided - SKIPPED")
            return False

        return True

    def _run(
        self,
        framework: Framework,
        format: Format,
        workspace: Workspace,
        path: pathlib.Path,
        batch_dim: Optional[int],
        dataloader: SizedDataLoader,
        verify_func: VerifyFunction,
        runner_cls: Type[NavigatorRunner],
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        model: Optional[Any] = None,
    ) -> CommandOutput:
        """Run verification.

        Args:
            framework (Framework): Model framework.
            format (Format): Model format.
            workspace (Path): Model Navigator workspace path.
            path (Path): Model path, relative to the workspace path.
            batch_dim: Batch dimension.
            dataloader (SizedDataLoader): Model dataloader.
            verify_func (VerifyFunction): Boolean function that verifies the runner based on outputs
                of the runner and source model.
            runner_cls (Type[NavigatorRunner]): Type of the runner to use for verification.
            input_metadata (TensorMetadata): Input metadata.
            output_metadata (TensorMetadata): Output metadata.
            model (Optional[Any], optional): Model if source model should be used. Defaults to None.

        Returns:
            CommandOutput: Status OK if succesfull verification, FAIL otherwise.
        """
        LOGGER.info(f"Verification for: {format} {runner_cls} started.")

        source_format = FRAMEWORK2BASE_FORMAT[framework]
        source_runners = get_source_default_runners(source_format)

        if runner_cls.name() in [runner.name() for runner in source_runners]:
            LOGGER.info("Runner is the same as source model.")
            return CommandOutput(status=CommandStatus.OK)

        def _get_outputs(runner):
            with runner:
                for sample in dataloader:
                    sample = extract_sample(sample, input_metadata, framework)
                    output = runner.infer(sample)
                    yield output

        runner = get_runner(runner_cls)(
            model=model if format == source_format else workspace.path / path,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            navigator_workspace=workspace.path,
            batch_dim=batch_dim,
        )
        y_pred = _get_outputs(runner)

        fw_runner = get_runner(source_runners[0])(
            model=model,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
        )
        y_fw = _get_outputs(fw_runner)
        is_verified = verify_func(y_pred, y_fw)
        return CommandOutput(status=CommandStatus.OK if is_verified is True else CommandStatus.FAIL)
