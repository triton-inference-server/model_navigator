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
"""JAX export."""

import pathlib
from typing import Optional

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.export import exporters
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.frameworks.jax import JaxModel
from model_navigator.utils.common import parse_kwargs_to_cmd


class ExportJAX2SavedModel(Command):
    """Export JAX to SavedModel command."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        jit_compile: bool,
        enable_xla: bool,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        verbose: bool,
        model: Optional[JaxModel] = None,
    ) -> CommandOutput:
        """Run export from JAX to SavedModel.

        For detailed explanation of jax2tf parameters please refer to [documentation]
        [documentation]: https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md

        Args:
            workspace (Path): Model Navigator workspace path.
            path (Path): Output SavedModel path relative to workspace path.
            jit_compile (bool): jax2tf parameter.
            enable_xla (bool): jax2tf parameter.
            input_metadata (TensorMetadata): Input metadata.
            output_metadata (TensorMetadata): Output metadata.
            verbose (bool): If True verbose logging.
            model (Optional[JaxModel], optional): JAX model. Defaults to None.

        Returns:
            CommandOutput: _description_
        """
        LOGGER.info(f"JAX to SavedModel export started {jit_compile=}, {enable_xla=}")

        exported_model_path = workspace.path / path
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return CommandOutput(status=CommandStatus.SKIPPED)

        assert model is not None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        exporters.jax2savedmodel.get_model = lambda: model.model
        exporters.jax2savedmodel.get_model_params = lambda: model.params

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "jit_compile": jit_compile,
                "enable_xla": enable_xla,
                "input_metadata": input_metadata.to_json(),
                "output_names": list(output_metadata.keys()),
                "navigator_workspace": workspace.path.as_posix(),
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_local_runtime_script(
                exporters.jax2savedmodel.__file__, exporters.jax2savedmodel.export, args
            )
        return CommandOutput(status=CommandStatus.OK)
