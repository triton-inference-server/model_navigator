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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from model_navigator.api.config import DeviceKind, JitType
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.export import exporters
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.execution_context import ExecutionContext
from model_navigator.logger import LOGGER
from model_navigator.utils.common import parse_kwargs_to_cmd
from model_navigator.utils.tensor import TensorMetadata


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
        workspace: Path,
        path: Path,
        target_device: DeviceKind,
        jit_type: JitType,
        verbose: bool,
        model: Optional[Any] = None,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            path: Path inside the workspace where exported model is stored
            verbose: Enable verbose logging
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes

        Returns:
            CommandOutput object with status
        """
        LOGGER.info("TorchScrip export started")

        exported_model_path = workspace / path
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
                "exported_model_path": exported_model_path.relative_to(workspace).as_posix(),
                "target_jit_type": jit_type.value,
                "batch_dim": batch_dim,
                "target_device": target_device.value,
                "navigator_workspace": workspace.as_posix(),
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
        workspace: Path,
        path: Path,
        opset: int,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: DeviceKind,
        verbose: bool,
        forward_kw_names: Optional[Tuple[str, ...]] = None,
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
            forward_kw_names: Additional arguments to override input names
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes
            dynamic_axes: Definition of model inputs dynamic axes

        Returns:
            CommandOutput object with status
        """
        LOGGER.info("PyTorch to ONNX export started")
        exported_model_path = workspace / path
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

        model.to(target_device.value)

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

            kwargs = {
                "navigator_workspace": workspace.as_posix(),
                "exported_model_path": exported_model_path.relative_to(workspace).as_posix(),
                "opset": opset,
                "input_names": list(input_metadata.keys()),
                "output_names": list(output_metadata.keys()),
                "dynamic_axes": dynamic_axes,
                "batch_dim": batch_dim,
                "forward_kw_names": list(forward_kw_names) if forward_kw_names else None,
                "target_device": target_device.value,
            }

            args = parse_kwargs_to_cmd(kwargs)
            context.execute_local_runtime_script(exporters.torch2onnx.__file__, exporters.torch2onnx.export, args)

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
