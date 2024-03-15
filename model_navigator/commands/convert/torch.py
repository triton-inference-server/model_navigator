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
"""TorchScript conversions."""

import pathlib
from typing import Any, Dict, List, Optional

from model_navigator.api.config import DeviceKind, TensorRTPrecision, TensorRTPrecisionMode, TensorRTProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.convert.base import Convert2TensorRTWithMaxBatchSizeSearch
from model_navigator.commands.convert.converters import ts2onnx, ts2torchtrt
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.utils import devices
from model_navigator.utils.common import parse_kwargs_to_cmd


class ConvertTorchScript2ONNX(Command):
    """Convert TorchScript to ONNX."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        parent_path: pathlib.Path,
        opset: int,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: DeviceKind,
        verbose: bool,
        custom_args: Dict[str, Any],
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Run TorchScript to ONNX conversion.

        Args:
            workspace (Path): Model Navigator workspace path.
            path (Path): Output ONNX model path relative to workspace path.
            parent_path (Path): TorchScript model path relative to workspace path.
            opset (int): ONNX opset.
            input_metadata (TensorMetadata): Input metadata.
            output_metadata (TensorMetadata): Output metadata.
            target_device (DeviceKind): Device to load TorchScript model on.
            verbose (bool): If True verbose logging.
            batch_dim (Optional[int], optional): Batch dimension. Defaults to None.
            custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.export.
                For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
        Returns:
            CommandOutput: Status OK.
        """
        LOGGER.info("TorchScript to ONNX conversion started")
        exported_model_path = workspace.path / parent_path
        converted_model_path = workspace.path / path
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        with ExecutionContext(
            workspace=workspace,
            script_path=converted_model_path.parent / "reproduce_conversion.py",
            cmd_path=converted_model_path.parent / "reproduce_conversion.sh",
            verbose=verbose,
        ) as context:
            kwargs = {
                "navigator_workspace": workspace.path.as_posix(),
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "converted_model_path": converted_model_path.relative_to(workspace.path).as_posix(),
                "opset": opset,
                "input_metadata": input_metadata.to_json(),
                "input_names": list(input_metadata.keys()),
                "output_names": list(output_metadata.keys()),
                "dynamic_axes": dict(**input_metadata.dynamic_axes, **output_metadata.dynamic_axes),
                "batch_dim": batch_dim,
                "target_device": target_device.value,
                "custom_args": custom_args,
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_external_runtime_script(ts2onnx.__file__, args)
        LOGGER.info("Converted TorchScript to ONNX.")
        return CommandOutput(status=CommandStatus.OK)


class ConvertTorchScript2TorchTensorRT(Convert2TensorRTWithMaxBatchSizeSearch):
    """Convert TorchScript to Torch-TensorRT."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        parent_path: pathlib.Path,
        precision: TensorRTPrecision,
        input_metadata: TensorMetadata,
        precision_mode: TensorRTPrecisionMode,
        max_workspace_size: int,
        target_device: DeviceKind,
        verbose: bool,
        debug: bool,
        dataloader_trt_profile: TensorRTProfile,
        custom_args: Dict[str, Any],
        batch_dim: Optional[int] = None,
        dataloader_max_batch_size: Optional[int] = None,
        device_max_batch_size: Optional[int] = None,
        optimized_trt_profiles: Optional[List[TensorRTProfile]] = None,
    ) -> CommandOutput:
        """Run Torchscript ot Torch-TensorRT conversion.

        For detailed explanation of TensorRT parameters please refer to [documentation]
        [documentation]: https://pytorch.org/TensorRT/

        Args:
            workspace (Path): Model Navigator workspace path.
            path (Path): Output Torch-TensorRT model path relative to workspace path.
            parent_path (Path): TorchScript model path relative to workspace path.
            precision (TensorRTPrecision): TensorRTPrecision.
            input_metadata (TensorMetadata): Input metadata.
            precision_mode (TensorRTPrecisionMode): TensorRT precision mode.
            max_workspace_size (int): TensorRT maximum workspace size.
            target_device (DeviceKind): Device to load TorchScript model on.
            verbose (bool): If True verbose logging.
            debug (bool): If True print debug logs.
            dataloader_trt_profile (TensorRTProfile): Dataloader TensorRT profile.
            custom_args (Dict[str, Any]): Custom arguments for conversion.
            batch_dim (Optional[int], optional): Batch dimension. Defaults to None.
            dataloader_max_batch_size (Optional[int], optional): Maximum batch size form the dataloader.
                Defaults to None.
            device_max_batch_size (Optional[int], optional): Device maximum batch size.
                Defaults to None.
            optimized_trt_profiles (Optional[TensorRTProfile], optional): User specified TensorRT profile. Defaults to None.

        Raises:
            RuntimeError: When no GPU is available.

        Returns:
            CommandOutput: Status OK.
        """
        LOGGER.info("Conversion TorchScript to TorchTensorRT started")
        if not devices.get_available_gpus():
            raise RuntimeError("No GPUs available.")

        exported_model_path = workspace.path / parent_path
        converted_model_path = workspace.path / path

        if not exported_model_path.exists():
            LOGGER.warning(f"Exported TorchScript model not found at {exported_model_path}. Skipping conversion.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        input_dtypes_str = [tensorrt_utils.cast_type(input_spec.dtype).name for input_spec in input_metadata.values()]

        def get_args(max_batch_size=None):
            # NOTE: Torch-TensorRT does not support multiple profiles. Use first profile.
            # TODO: Add support for multiple profiles when Torch-TensorRT supports it.
            profile = optimized_trt_profiles[0] if optimized_trt_profiles else dataloader_trt_profile
            shapes = self._get_shape_args(trt_profile=profile, batch_dim=batch_dim, max_batch_size=max_batch_size)

            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "converted_model_path": converted_model_path.relative_to(workspace.path).as_posix(),
                "shapes": shapes,
                "input_dtypes": input_dtypes_str,
                "max_workspace_size": max_workspace_size,
                "precision": precision.value,
                "precision_mode": precision_mode.value,
                "target_device": target_device.value,
                "navigator_workspace": workspace.path.as_posix(),
                "custom_args": custom_args,
                "debug": debug,
            }
            args = parse_kwargs_to_cmd(kwargs)
            return args

        with ExecutionContext(
            workspace=workspace,
            script_path=converted_model_path.parent / "reproduce_conversion.py",
            cmd_path=converted_model_path.parent / "reproduce_conversion.sh",
            verbose=verbose,
        ) as context:
            conversion_max_batch_size = self._execute_conversion(
                convert_func=lambda args: context.execute_external_runtime_script(ts2torchtrt.__file__, args),
                get_args=get_args,
                batch_dim=batch_dim,
                device_max_batch_size=device_max_batch_size,
                dataloader_max_batch_size=dataloader_max_batch_size,
                custom_trt_profile_available=bool(optimized_trt_profiles),
            )

        conversion_profiles = self._get_shape_args(
            trt_profile=optimized_trt_profiles[0] if optimized_trt_profiles else dataloader_trt_profile,
            batch_dim=batch_dim,
            max_batch_size=conversion_max_batch_size,
        )
        LOGGER.info("Converted TorchScript to Torch-TensorRT.")
        return CommandOutput(
            status=CommandStatus.OK,
            output={"conversion_max_batch_size": conversion_max_batch_size, "conversion_profiles": conversion_profiles},
        )

    @staticmethod
    def _get_shape_args(
        trt_profile: TensorRTProfile,
        batch_dim: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        if batch_dim is not None and max_batch_size is not None and max_batch_size > 0:
            trt_profile = tensorrt_utils.get_trt_profile_with_new_max_batch_size(
                trt_profile=trt_profile,
                max_batch_size=max_batch_size,
                batch_dim=batch_dim,
            )

        shape_args = {name: vars(shape_tuple) for name, shape_tuple in trt_profile.items()}
        return shape_args
