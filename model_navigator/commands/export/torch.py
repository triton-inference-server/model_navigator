# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
from typing import Any, Dict, List, Optional, Union, cast

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.export import exporters
from model_navigator.configuration import DeviceKind, JitType, TensorRTPrecision, TensorRTProfile
from model_navigator.configuration.model.model_config import OnnxDynamoExportConfig, ONNXExportEngine
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.frameworks.torch.utils import offload_torch_model_to_cpu
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
        model: Any = None,
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
        LOGGER.info("TorchScript export started")

        exported_model_path = workspace.path / path
        if exported_model_path.is_file() or exported_model_path.is_dir():
            LOGGER.info("Model already exists. Skipping export.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected model of type torch.nn.Module. Got None instead.")

        exporters.torch2torchscript.get_model = lambda: model

        # Keep model on CPU after operation
        def on_exit():
            offload_torch_model_to_cpu(model)

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

            context.execute_python_script(
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
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], list]]] = None,
        custom_args: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        batch_dim: Optional[int] = None,
        export_device: Optional[str] = None,
        export_engine: Optional[ONNXExportEngine] = None,
        # Torch Dynamo path (injected)
        dataloader_trt_profile: Optional[TensorRTProfile] = None,
        dataloader_max_batch_size: Optional[int] = None,
        device_max_batch_size: Optional[int] = None,
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
            dynamic_axes: Definition of model inputs dynamic axes
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes
            export_device: Device to export model to ONNX on. If None use target_device.
            export_engine: Export engine config to use
            custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.export
                For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export

            dataloader_trt_profile: TensorRT profile for dataloader (injected)
            dataloader_max_batch_size: The maximal batch size obtained from dataloader
            device_max_batch_size: The maximal batch size obtained for device

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

        exporters.torch2onnx.get_model = lambda: model

        device = export_device or target_device.value

        if dynamic_axes is None:
            dynamic_axes = dict(**input_metadata.dynamic_axes, **output_metadata.dynamic_axes)
            LOGGER.warning(f"No dynamic axes provided. Using values derived from the dataloader: {dynamic_axes}")
        else:
            _validate_if_dynamic_axes_aligns_with_dataloader_shapes(dynamic_axes, input_metadata, output_metadata)

        # Keep model on CPU after operation
        def on_exit():
            offload_torch_model_to_cpu(model)

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
            on_exit=on_exit,
        ) as context:
            kwargs = {
                "export_engine": "torch-trace",
                "navigator_workspace": workspace.path.as_posix(),
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "opset": opset,
                "input_metadata": input_metadata.to_json(),
                "input_names": list(input_metadata.keys()),
                "output_names": list(output_metadata.keys()),
                "batch_dim": batch_dim,
                "dynamic_axes": dynamic_axes,
                "target_device": device,
                "custom_args": custom_args,
                "verbose": verbose,
            }

            # override kwargs if dynamo export engine is used
            if isinstance(export_engine, OnnxDynamoExportConfig):
                assert dataloader_trt_profile is not None, "Dataloader TRT profile is required for Torch Dynamo export"

                export_engine = cast(OnnxDynamoExportConfig, export_engine)

                dynamic_shapes = batch_dim is not None or dynamic_axes
                if export_engine.dynamo_dynamic_shapes is not None:
                    dynamic_shapes = export_engine.dynamo_dynamic_shapes

                kwargs["export_engine"] = "torch-dynamo"
                kwargs["dynamic_shapes"] = dynamic_shapes
                kwargs["dataloader_max_batch_size"] = dataloader_max_batch_size
                kwargs["device_max_batch_size"] = device_max_batch_size
                kwargs["dataloader_trt_profile"] = dataloader_trt_profile.to_dict()

            args = parse_kwargs_to_cmd(kwargs)
            context.execute_python_script(exporters.torch2onnx.__file__, exporters.torch2onnx.export, args)

        return CommandOutput(status=CommandStatus.OK)


class ExportExportedProgram(Command):
    """Command for exporting Torch models to ExportedProgram."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        dataloader_trt_profile: TensorRTProfile,
        target_device: DeviceKind,
        verbose: bool,
        custom_args: Dict[str, Any],
        model: Any = None,
        batch_dim: Optional[int] = None,
        dataloader_max_batch_size: Optional[int] = None,
        device_max_batch_size: Optional[int] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            path: Path inside the workspace where exported model is stored
            input_metadata: Model inputs metadata
            output_metadata: Model outputs metadata
            dataloader_trt_profile: Profile from dataloader
            target_device: Target device for export - determine the exported model
            verbose: Enable verbose logging
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes
            custom_args: Passthrough parameters for torch.onnx.export
                For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
            dataloader_max_batch_size: The maximal batch size obtained from datalaoder
            device_max_batch_size: The maximal batch size obtained for device

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

        exporters.torch2exportedprogram.get_model = lambda: model

        # Keep model on CPU after operation
        def on_exit():
            offload_torch_model_to_cpu(model)

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
                "dataloader_trt_profile": dataloader_trt_profile.to_dict(),
                "batch_dim": batch_dim,
                "target_device": target_device.value,
                "navigator_workspace": workspace.path.as_posix(),
                "dataloader_max_batch_size": dataloader_max_batch_size,
                "device_max_batch_size": device_max_batch_size,
                "custom_args": custom_args,
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_python_script(
                exporters.torch2exportedprogram.__file__, exporters.torch2exportedprogram.export, args
            )

        return CommandOutput(status=CommandStatus.OK)


class ExportOnnxFromQuantizedTorch(Command):
    """Command for exporting a PyTorch model to a quantized ONNX using ModelOpt.

    This command first runs quantization via ModelOpt and then exports the quantized model to ONNX.
    API reference:
      - mtq.model_quant: https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html
      - mtq.export_onnx: https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.export_onnx.html
    """

    def _run(
        self,
        workspace: Workspace,
        opset: int,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        precision: TensorRTPrecision,
        target_device: Any,  # expected to have a .value attribute
        verbose: bool,
        custom_args: Dict[str, Any],
        model: Any = None,
        batch_dim: Optional[int] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], list]]] = None,
        export_device: Optional[str] = None,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            opset: ONNX opset
            input_metadata: Model inputs metadata
            output_metadata: Model outputs metadata
            precision: TensorRT precision to use for the model
            target_device: Target device for export
            verbose: Enable verbose logging
            custom_args: Passthrough parameters for torch.onnx.export
            model: The model that has to be exported
            batch_dim: Location of batch position in shapes
            dynamic_axes: Definition of model inputs dynamic axes
            export_device: Device to export model to ONNX on. If None use target_device.

        Returns:
            CommandOutput object with status
        """
        LOGGER.info("ExportOnnxFromQuantizedTorch started")

        exported_model_path = workspace.path / f"trt-{precision.value}" / "quantized_model.onnx"
        if exported_model_path.exists():
            LOGGER.info("Quantized ONNX model already exists. Skipping export.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected a torch.nn.Module model. Got None.")

        # Set up the device to use
        device = export_device or target_device.value

        # Set up the get_model function for the exporter script
        from model_navigator.commands.export.exporters import torch2quantized_onnx

        torch2quantized_onnx.get_model = lambda: model

        # Keep model on CPU after operation
        def on_exit():
            offload_torch_model_to_cpu(model)

        with ExecutionContext(
            workspace=workspace,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
            on_exit=on_exit,
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "opset": opset,
                "input_metadata": input_metadata.to_json(),
                "output_names": list(output_metadata.keys()),
                "dynamic_axes": dynamic_axes,
                "batch_dim": batch_dim,
                "target_device": device,
                "precision": precision.value,
                "custom_args": custom_args,
                "navigator_workspace": workspace.path.as_posix(),
                "verbose": verbose,
            }

            args = parse_kwargs_to_cmd(kwargs)

            try:
                context.execute_python_script(torch2quantized_onnx.__file__, torch2quantized_onnx.export, args)
                return CommandOutput(status=CommandStatus.OK)
            except Exception as e:
                LOGGER.error(f"Error during quantized ONNX export: {str(e)}")
                return CommandOutput(status=CommandStatus.FAIL)


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
