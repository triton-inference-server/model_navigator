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
"""Torch based formats conversions."""

import pathlib
from typing import Any, Dict, List, Optional

import model_navigator.core.context as ctx
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.convert.base import Convert2TensorRTWithMaxBatchSizeSearch
from model_navigator.commands.convert.converters import ep2torchtrt, ts2onnx
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.configuration import DeviceKind, TensorRTPrecision, TensorRTPrecisionMode, TensorRTProfile
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.frameworks.tensorrt.timing_tactics import trt_cache_inplace_cache_dir
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
            workspace: Model Navigator workspace path.
            path: Output ONNX model path relative to workspace path.
            parent_path: TorchScript model path relative to workspace path.
            opset: ONNX opset.
            input_metadata: Input metadata.
            output_metadata: Output metadata.
            target_device: Device to load TorchScript model on.
            verbose: If True verbose logging.
            batch_dim: Batch dimension. Defaults to None.
            custom_args: Passthrough parameters for torch.onnx.export. For available arguments check PyTorch
                         documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
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

            context.execute_python_script(
                ts2onnx.__file__,
                ts2onnx.convert,
                args,
                run_in_isolation=True,
            )
        LOGGER.info("Converted TorchScript to ONNX.")
        return CommandOutput(status=CommandStatus.OK)


class ConvertExportedProgram2TorchTensorRT(Convert2TensorRTWithMaxBatchSizeSearch):
    """Convert ExportedProgram to Torch-TensorRT."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        parent_path: pathlib.Path,
        input_metadata: TensorMetadata,
        target_device: DeviceKind,
        precision: TensorRTPrecision,
        precision_mode: TensorRTPrecisionMode,
        max_workspace_size: int,
        pickle_protocol: int,
        verbose: bool,
        debug: bool,
        dataloader_trt_profile: TensorRTProfile,
        custom_args: Dict[str, Any],
        conversion_fallback: bool = False,
        batch_dim: Optional[int] = None,
        dataloader_max_batch_size: Optional[int] = None,
        device_max_batch_size: Optional[int] = None,
        trt_profiles: Optional[List[TensorRTProfile]] = None,
    ) -> CommandOutput:
        """Run ExportedProgram ot Torch-TensorRT conversion.

        For detailed explanation of TensorRT parameters please refer to [documentation]
        [documentation]: https://pytorch.org/TensorRT/

        Args:
            workspace: Model Navigator workspace path.
            path: Output Torch-TensorRT model path relative to workspace path.
            parent_path: ExportedProgram model path relative to workspace path.
            input_metadata: Input metadata.
            target_device: Device on which run conversion.
            precision: TensorRTPrecision.
            precision_mode: TensorRT precision mode.
            max_workspace_size: TensorRT maximum workspace size.
            pickle_protocol: Pickle protocol for model serialization.
            verbose: If True verbose logging.
            debug: If True print debug logs.
            dataloader_trt_profile: Dataloader TensorRT profile.
            custom_args: Custom arguments for conversion.
            conversion_fallback: Enable fallback for conversion to try conversion with smaller batch size
            batch_dim : Batch dimension. Defaults to None.
            dataloader_max_batch_size: Maximum batch size form the dataloader. Defaults to None.
            device_max_batch_size: Device maximum batch size. Defaults to None.
            trt_profiles: User specified TensorRT profile. Defaults to None.

        Raises:
            RuntimeError: When no GPU is available.

        Returns:
            CommandOutput: Status OK.
        """
        LOGGER.info("Conversion ExportedProgram to TorchTensorRT started")
        if not devices.get_available_gpus():
            raise RuntimeError("No GPUs available.")

        exported_model_path = workspace.path / parent_path
        converted_model_path = workspace.path / path

        if not exported_model_path.exists():
            LOGGER.warning(f"ExportedProgram model not found at {exported_model_path}. Skipping conversion.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        def get_args(max_batch_size=None):
            # NOTE: Torch-TensorRT does not support multiple profiles. Use first profile.
            # TODO: Add support for multiple profiles when Torch-TensorRT supports it.
            profile = trt_profiles[0] if trt_profiles else dataloader_trt_profile
            shapes = self._get_shape_args(trt_profile=profile, batch_dim=batch_dim, max_batch_size=max_batch_size)
            module_name = ctx.global_context.get(ctx.INPLACE_OPTIMIZE_MODULE_NAME_CONTEXT_KEY) or workspace.path.stem

            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace.path).as_posix(),
                "converted_model_path": converted_model_path.relative_to(workspace.path).as_posix(),
                "input_metadata": input_metadata.to_json(),
                "shapes": shapes,
                "batch_dim": batch_dim,
                "max_workspace_size": max_workspace_size,
                "precision": precision.value,
                "precision_mode": precision_mode.value,
                "pickle_protocol": pickle_protocol,
                "navigator_workspace": workspace.path.as_posix(),
                "target_device": target_device.value,
                "custom_args": custom_args,
                "model_name": module_name,
                "timing_cache_dir": trt_cache_inplace_cache_dir(),
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
                convert_func=lambda args: context.execute_python_script(
                    ep2torchtrt.__file__,
                    ep2torchtrt.convert,
                    args,
                    run_in_isolation=True,
                ),
                get_args=get_args,
                batch_dim=batch_dim,
                device_max_batch_size=device_max_batch_size,
                dataloader_max_batch_size=dataloader_max_batch_size,
                custom_trt_profile_available=bool(trt_profiles),
                conversion_fallback=conversion_fallback,
            )

        conversion_profiles = self._get_shape_args(
            trt_profile=trt_profiles[0] if trt_profiles else dataloader_trt_profile,
            batch_dim=batch_dim,
            max_batch_size=conversion_max_batch_size,
        )
        LOGGER.info("Converted ExportedProgram to Torch-TensorRT.")
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
