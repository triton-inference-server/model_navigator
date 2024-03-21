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
"""Commands for exporting Tensorflow models."""

import pathlib

import tensorflow as tf  # pytype: disable=import-error

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.export import exporters
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.utils.common import parse_kwargs_to_cmd


class ExportTF2SavedModel(Command):
    """Tensorflow to SavedModel exporter."""

    def _run(
        self,
        model: tf.keras.Model,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        workspace: Workspace,
        path: pathlib.Path,
        verbose: bool,
    ) -> CommandOutput:
        """Run Tensorflow to SavedModel export.

        Args:
            model (tf.keras.Model): Keras model to be exported.
            input_metadata (TensorMetadata): Input metadata.
            output_metadata (TensorMetadata): Output metadata.
            workspace (Path): Model Navigator workspace path.
            path (Path): Output SavedModel path relative to workspace path.
            verbose (bool): If True verbose logging.

        Returns:
            CommandOutput: Status OK.
        """
        LOGGER.info("TensorFlow2 to SavedModel export started")

        exported_model_path = workspace.path / path
        if exported_model_path.exists():
            LOGGER.info("Model already exists. Skipping export.")
            return CommandOutput(status=CommandStatus.SKIPPED)
        assert model is not None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        exporters.keras2savedmodel.get_model = lambda: model

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
                "navigator_workspace": workspace.path.as_posix(),
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_local_runtime_script(
                exporters.keras2savedmodel.__file__, exporters.keras2savedmodel.export, args
            )

        return CommandOutput(status=CommandStatus.OK)


class UpdateSavedModelSignature(Command):
    """SavedModel signature udpater."""

    def _run(
        self,
        path: pathlib.Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        workspace: Workspace,
        verbose: bool,
    ) -> CommandOutput:
        """Update SavedModel signature so it matches IO metadata.

        Args:
            path (Path): SavedModel path relative to workspace path.
            input_metadata (TensorMetadata): Input metadata.
            output_metadata (TensorMetadata): Output metadata.
            workspace (Path): Model Navigator workspace path.
            verbose (bool): If True verbose logging.

        Returns:
            CommandOutput: Status OK.
        """
        LOGGER.info("TensorFlow2 to SavedModel export started")

        exported_model_path = workspace.path / path
        assert exported_model_path.exists()
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        exporters.savedmodel2savedmodel.get_model = lambda: tf.keras.models.load_model(  # pytype: disable=module-attr
            exported_model_path
        )

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "input_metadata": input_metadata.to_json(),
                "output_names": list(output_metadata.keys()),
                "navigator_workspace": workspace.path.as_posix(),
            }

            args = parse_kwargs_to_cmd(kwargs)
            context.execute_local_runtime_script(
                exporters.savedmodel2savedmodel.__file__, exporters.savedmodel2savedmodel.update_signature, args
            )

        return CommandOutput(status=CommandStatus.OK)
