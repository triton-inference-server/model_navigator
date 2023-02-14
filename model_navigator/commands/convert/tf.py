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
"""Saved Model conversions."""
from pathlib import Path
from typing import Optional

from model_navigator.api.config import TensorRTPrecision, TensorRTProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.convert.base import Convert2TensorRTWithMaxBatchSizeSearch
from model_navigator.commands.convert.converters import sm2tftrt
from model_navigator.execution_context import ExecutionContext
from model_navigator.logger import LOGGER
from model_navigator.utils import devices
from model_navigator.utils.common import parse_kwargs_to_cmd


class ConvertSavedModel2ONNX(Command):
    """Convert SavedModel to ONNX."""

    def _run(
        self,
        workspace: Path,
        path: Path,
        parent_path: Path,
        opset: int,
        verbose: bool,
    ) -> CommandOutput:
        """Run Conversion from SavedModel to ONNX.

        Args:
            workspace (Path): Navigator workspace.
            path (Path): ONNX target path.
            parent_path (Path): Saved Model path.
            opset (int): ONNX opset.
            verbose (bool): If True verbose logging.

        Returns:
            CommandOutput: Conversion output.
        """
        LOGGER.info("SavedModel to ONNX conversion started")

        exported_model_path = workspace / parent_path
        converted_model_path = workspace / path
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)
        if converted_model_path.exists():
            LOGGER.info("Model already exists. Skipping conversion.")
            return CommandOutput(status=CommandStatus.SKIPPED)
        if not exported_model_path.exists():
            LOGGER.warning(f"Exported SavedModel model not found at {exported_model_path}. Skipping conversion")
            return CommandOutput(status=CommandStatus.SKIPPED)

        convert_cmd = [
            "python",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            exported_model_path.relative_to(workspace).as_posix(),
            "--output",
            converted_model_path.relative_to(workspace).as_posix(),
            "--opset",
            str(opset),
        ]

        with ExecutionContext(
            workspace=workspace,
            cmd_path=converted_model_path.parent / "reproduce_conversion.sh",
            verbose=verbose,
        ) as context:
            context.execute_cmd(convert_cmd)
        LOGGER.info("Converted SavedModel to ONNX.")
        return CommandOutput(status=CommandStatus.OK)


class ConvertSavedModel2TFTRT(Convert2TensorRTWithMaxBatchSizeSearch):
    """Convert SavedModel to Tensorflow-TensorRT."""

    def _run(
        self,
        max_workspace_size: int,
        minimum_segment_size: int,
        workspace: Path,
        path: Path,
        parent_path: Path,
        precision: TensorRTPrecision,
        verbose: bool,
        dataloader_trt_profile: TensorRTProfile,
        batch_dim: Optional[int] = None,
        dataloader_max_batch_size: Optional[int] = None,
        device_max_batch_size: Optional[int] = None,
        trt_profile: Optional[TensorRTProfile] = None,
    ) -> CommandOutput:
        """Run conversion from SavedModel to Tensorflow-TensorRT.

        Args:
            max_workspace_size (int): TRT max workspaze size in bytes.
            minimum_segment_size (int): TRT minimu segment size.
            workspace (Path): navigator workspace.
            path (Path): Tensorflow-TensorRT target path.
            parent_path (Path): SavedModel path.
            precision (TensorRTPrecision): Tensorflow-TensorRT model precision.
            verbose (bool): If True verbose logging.
            dataloader_trt_profile (TensorRTProfile): Dataloader TensorRT profile.
            batch_dim (Optional[int], optional): Batching axis.. Defaults to None.
            dataloader_max_batch_size (Optional[int], optional): Maximum batch size from the dataloader.
                Defaults to None.
            device_max_batch_size (Optional[int], optional): Maximu batch size that fits on the device.
                Defaults to None.
            trt_profile (Optional[TensorRTProfile], optional): User specified TensorRT profile. Defaults to None.

        Raises:
            RuntimeError: When no GPUs are available.

        Returns:
            CommandOutput: Conversion output.
        """
        LOGGER.info("SavedModel to Tensorflow-TensorRT conversion started")
        if not devices.get_available_gpus():
            raise RuntimeError("No GPUs available.")

        exported_model_path = workspace / parent_path
        converted_model_path = workspace / path
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        if converted_model_path.exists():
            LOGGER.info("Model already exists. Skipping conversion.")
            return CommandOutput(status=CommandStatus.SKIPPED)
        if not exported_model_path.exists():
            LOGGER.warning(f"Exported SavedModel model not found at {exported_model_path}. Skipping conversion")
            return CommandOutput(status=CommandStatus.SKIPPED)

        custom_trt_profile = trt_profile
        trt_profile = self._get_trt_profile(
            dataloader_trt_profile=dataloader_trt_profile, custom_trt_profile=custom_trt_profile
        )

        if batch_dim is not None:
            max_batch_size = list(trt_profile.values())[0].max[batch_dim]
        else:
            max_batch_size = None

        if trt_profile is not None:
            LOGGER.info("TF-TensorRT conversion currently does not support custom profiles.")

        def get_args(max_batch_size: int = max_batch_size):
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workspace).as_posix(),
                "converted_model_path": converted_model_path.relative_to(workspace).as_posix(),
                "max_workspace_size": max_workspace_size,
                "target_precision": precision.value,
                "minimum_segment_size": minimum_segment_size,
                "batch_dim": batch_dim,
                "navigator_workspace": workspace.as_posix(),
                "max_batch_size": max_batch_size,
            }

            args = parse_kwargs_to_cmd(kwargs)
            return args

        with ExecutionContext(
            workspace=workspace,
            script_path=converted_model_path.parent / "reproduce_conversion.py",
            cmd_path=converted_model_path.parent / "reproduce_conversion.sh",
            verbose=verbose,
        ) as context:

            self._execute_conversion(
                convert_func=lambda args: context.execute_external_runtime_script(sm2tftrt.__file__, args),
                get_args=get_args,
                batch_dim=batch_dim,
                device_max_batch_size=device_max_batch_size,
                dataloader_max_batch_size=dataloader_max_batch_size,
                custom_trt_profile_available=bool(custom_trt_profile),
            )
        LOGGER.info("Converted SavedModel to Tensorflow-TensorRT.")

        return CommandOutput(status=CommandStatus.OK)
