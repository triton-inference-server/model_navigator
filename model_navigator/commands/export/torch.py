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
"""Commands for exporting PyTorch model from source to serialized formats.

The module provide functionality to export model to TorchScript and/or ONNX.
"""

import pathlib
from typing import Any, Dict, List, Optional, Union

from model_navigator.api.config import DeviceKind, JitType
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.export import exporters
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.utils.common import parse_kwargs_to_cmd


class ExportTorch2TorchScript(Command):
    """Command for export PyTorch model to TorchScript.

    Example of use:
        ExportTorch2TorchScript.run(
            workspace="/path/to/working/directory",
            path="/path/inside/workdir/to/store/exported/model",
            target_device=DeviceKind.CUDA,
            jit_type=JitType.TRACE,
            verbose=True,
            model=torch.nn.Identity(),
            batch_dim=[0]
        )
    """

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        target_device: DeviceKind,
        jit_type: JitType,
        input_metadata: TensorMetadata,
        verbose: bool,
        strict: bool,
        custom_args: Dict[str, Any],
        model: Optional[Any] = None,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            path: Path inside the workspace where exported model is stored.
            target_device: Device to load TorchScript model on.
            jit_type: TorchScript jit type.
            input_metadata: Model inputs metadata
            verbose: Enable verbose logging.
            strict: Enable or Disable strict flag for tracer used in TorchScript export.
            model: The model that has to be exported.
            batch_dim: Location of batch position in shapes.
            custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.jit.trace
                For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/jit.html#torch.jit.trace
        Returns:
            CommandOutput object with status
        """
        LOGGER.info("TorchScrip export started")

        exported_model_path = workspace.path / path
        if exported_model_path.is_file() or exported_model_path.is_dir():
            LOGGER.info("Model already exists. Skipping export.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected model of type torch.nn.Module. Got None instead.")

        model.to(target_device.value)

        exporters.torch2torchscript.get_model = lambda: model

        # Keep model on CPU after operation
        def on_exit():
            model.to("cpu")

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
            on_exit=on_exit,
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "target_jit_type": jit_type.value,
                "input_metadata": input_metadata.to_json(),
                "batch_dim": batch_dim,
                "target_device": target_device.value,
                "strict": strict,
                "navigator_workspace": workspace.path.as_posix(),
                "custom_args": custom_args,
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_local_runtime_script(
                exporters.torch2torchscript.__file__, exporters.torch2torchscript.export, args
            )

        return CommandOutput(status=CommandStatus.OK)


class ExportTorch2ONNX(Command):
    """Command for export PyTorch model to ONNX.

    Example of use:

        ExportTorch2ONNX.run(
            workspace="/path/to/working/directory",
            path="/path/inside/workdir/to/store/exported/model",
            opset=13,
            input_metadata={"input_1": TensorSpec(name="input_1", shape=(128, 20), dtype=np.dtype("float32")),
            output_metadata={"output_1": TensorSpec(name="output_1", shape=(128, 20), dtype=np.dtype("float32")),
            target_device=DeviceKind.CUDA,
            jit_type=JitType.TRACE,
            verbose=True,
            model=torch.nn.Identity(),
            batch_dim=[0]
        )
    """

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        opset: int,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: DeviceKind,
        verbose: bool,
        custom_args: Dict[str, Any],
        model: Optional[Any] = None,
        batch_dim: Optional[int] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        export_device: Optional[str] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            path: Path inside the workspace where exported model is stored
            opset: ONNX opset
            input_metadata: Model inputs metadata
            output_metadata: Model outputs metadata
            target_device: Target device for export - determine the exported model
            verbose: Enable verbose logging
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes
            dynamic_axes: Definition of model inputs dynamic axes
            custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.export
                For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
            export_device: Device to export model to ONNX on. If None use target_device.

        Returns:
            CommandOutput object with status
        """
        LOGGER.info("PyTorch to ONNX export started")
        exported_model_path = workspace.path / path
        if exported_model_path.exists():
            LOGGER.info("Model already exists. Skipping export.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected model of type torch.nn.Module. Got None instead.")

        if dynamic_axes is None:
            dynamic_axes = dict(**input_metadata.dynamic_axes, **output_metadata.dynamic_axes)
            LOGGER.warning(f"No dynamic axes provided. Using values derived from the dataloader: {dynamic_axes}")
        else:
            _validate_if_dynamic_axes_aligns_with_dataloader_shapes(dynamic_axes, input_metadata, output_metadata)

        exporters.torch2onnx.get_model = lambda: model

        # Keep model on CPU after operation
        def on_exit():
            model.to("cpu")

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
            on_exit=on_exit,
        ) as context:
            # import json # TODO fix that

            kwargs = {
                "navigator_workspace": workspace.path.as_posix(),
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "opset": opset,
                "input_metadata": input_metadata.to_json(),
                "input_names": list(input_metadata.keys()),
                "output_names": list(output_metadata.keys()),
                "dynamic_axes": dynamic_axes,
                "batch_dim": batch_dim,
                "export_device": export_device or target_device.value,
                "custom_args": custom_args,
            }

            args = parse_kwargs_to_cmd(kwargs)
            context.execute_local_runtime_script(exporters.torch2onnx.__file__, exporters.torch2onnx.export, args)

        return CommandOutput(status=CommandStatus.OK)


class ExportExportedProgram(Command):
    """Command for exporting Torch models to ExportedProgram."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: DeviceKind,
        verbose: bool,
        custom_args: Dict[str, Any],
        model: Optional[Any] = None,
        batch_dim: Optional[int] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            path: Path inside the workspace where exported model is stored
            opset: ONNX opset
            input_metadata: Model inputs metadata
            output_metadata: Model outputs metadata
            target_device: Target device for export - determine the exported model
            verbose: Enable verbose logging
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes
            dynamic_axes: Definition of model inputs dynamic axes
            custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.export
                For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export

        Returns:
            CommandOutput object with status
        """
        LOGGER.info("PyTorch ExportedProgram export started")

        exported_model_path = workspace.path / path
        if exported_model_path.is_file() or exported_model_path.is_dir():
            LOGGER.info("Model already exists. Skipping export.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected model of type torch.nn.Module. Got None instead.")

        model.to(target_device.value)

        exporters.torch2exportedprogram.get_model = lambda: model

        # Keep model on CPU after operation
        def on_exit():
            model.to("cpu")

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
            on_exit=on_exit,
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "input_metadata": input_metadata.to_json(),
                "batch_dim": batch_dim,
                "target_device": target_device.value,
                "navigator_workspace": workspace.path.as_posix(),
                "custom_args": custom_args,
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_local_runtime_script(
                exporters.torch2exportedprogram.__file__, exporters.torch2exportedprogram.export, args
            )

        return CommandOutput(status=CommandStatus.OK)


class ExportTorch2DynamoONNX(Command):
    """Command for exporting Torch models to ONNX with dynamo."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        input_metadata: TensorMetadata,
        target_device: DeviceKind,
        verbose: bool,
        custom_args: Dict[str, Any],
        model: Optional[Any] = None,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            path: Path inside the workspace where exported model is stored
            opset: ONNX opset
            input_metadata: Model inputs metadata
            target_device: Target device for export - determine the exported model
            verbose: Enable verbose logging
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes
            custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.dynamo_export
                Can be used to pass ExportOptions object.
                For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export

        Returns:
            CommandOutput object with status
        """
        LOGGER.info("PyTorch ExportedProgram export started")

        exported_model_path = workspace.path / path
        if exported_model_path.is_file() or exported_model_path.is_dir():
            LOGGER.info("Model already exists. Skipping export.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected model of type torch.nn.Module. Got None instead.")

        model.to(target_device.value)

        exporters.torch2dynamo_onnx.get_model = lambda: model

        # Keep model on CPU after operation
        def on_exit():
            model.to("cpu")

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
            on_exit=on_exit,
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "input_metadata": input_metadata.to_json(),
                "batch_dim": batch_dim,
                "target_device": target_device.value,
                "navigator_workspace": workspace.path.as_posix(),
                "custom_args": custom_args,
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_local_runtime_script(
                exporters.torch2dynamo_onnx.__file__, exporters.torch2dynamo_onnx.export, args
            )

        return CommandOutput(status=CommandStatus.OK)


def _validate_if_dynamic_axes_aligns_with_dataloader_shapes(
    dynamic_axes: Dict[str, Union[Dict[int, str], List[int]]],
    input_metadata: TensorMetadata,
    output_metadata: TensorMetadata,
):
    for name, axes in dynamic_axes.items():
        axes = list(axes)
        tensor_spec = input_metadata.get(name, None) or output_metadata.get(name, None)
        if tensor_spec is None:
            raise ModelNavigatorConfigurationError(f"Dynamic axis {axes} is specified for unknown input {name}.")
        for ax, d in enumerate(tensor_spec.shape):
            if d == -1 and ax not in axes:
                raise ModelNavigatorConfigurationError(
                    f"In tensor `{name}` axis `{ax}` is not set as dynamic axes but is dynamic in the dataloader."
                )
