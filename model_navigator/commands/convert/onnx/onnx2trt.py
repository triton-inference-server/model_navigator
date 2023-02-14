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
"""ConvertONNX2TRT command."""

import sys
from distutils.version import LooseVersion
from pathlib import Path
from typing import Optional

from model_navigator.api.config import TensorRTPrecision, TensorRTPrecisionMode, TensorRTProfile
from model_navigator.commands.base import CommandOutput, CommandStatus
from model_navigator.commands.convert.base import Convert2TensorRTWithMaxBatchSizeSearch
from model_navigator.execution_context import ExecutionContext
from model_navigator.logger import LOGGER
from model_navigator.runners.onnx import OnnxrtCPURunner
from model_navigator.runners.tensorrt import TensorRTRunner
from model_navigator.utils import devices, tensorrt
from model_navigator.utils.common import parse_kwargs_to_cmd
from model_navigator.utils.tensor import TensorMetadata


class ConvertONNX2TRT(Convert2TensorRTWithMaxBatchSizeSearch):
    """Command that converts ONNX checkpoint to TensorRT model plan."""

    def _run(
        self,
        workspace: Path,
        path: Path,
        parent_path: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        precision: TensorRTPrecision,
        precision_mode: TensorRTPrecisionMode,
        dataloader_trt_profile: TensorRTProfile,
        max_workspace_size: Optional[int] = None,
        batch_dim: Optional[int] = None,
        dataloader_max_batch_size: Optional[int] = None,
        device_max_batch_size: Optional[int] = None,
        trt_profile: Optional[TensorRTProfile] = None,
        verbose: bool = False,
    ) -> CommandOutput:
        """Run the ConvertONNX2TRT Command.

        Args:
            workspace (Path): Model Navigator working directory.
            path (Path): ONNX checkpoint path, relative to workspace.
            parent_path (Path): Path of ONNX parent model, relative to workspace.
            input_metadata (TensorMetadata): Model input metadata.
            output_metadata (TensorMetadata): Model output metadata.
            precision (TensorRTPrecision): TensoRT precision.
            precision_mode (TensorRTPrecisionMode): TensorRT precision mode.
            dataloader_trt_profile (TensorRTProfile): Dataloader TensorRT profile.
            max_workspace_size (Optional[int], optional): Maximum TensoRT workspace size, in bytes. Defaults to None.
            batch_dim (Optional[int], optional): Dimension of the batching, None if model does not support batching.
                Defaults to None.
            dataloader_max_batch_size (Optional[int], optional): Maximum batch size in the dataloader. Defaults to None.
            device_max_batch_size (Optional[int], optional): Maximum batch size that fits on the device.
                Defaults to None.
            trt_profile (Optional[TensorRTProfile], optional): User specified TensorRT profile. Defaults to None.
            verbose: enable verbose logging for command

        Returns:
            CommandOutput: Status and results of the command.
        """
        LOGGER.info("ONNX to TRT conversion started")
        if not devices.get_available_gpus():
            raise RuntimeError("No GPUs available.")

        input_model_path = workspace / parent_path
        converted_model_path = workspace / path
        converted_model_temp_path = converted_model_path.with_suffix(".temp")

        if converted_model_path.exists():
            LOGGER.info("Model already exists. Skipping conversion.")
            return CommandOutput(status=CommandStatus.SKIPPED)
        if not input_model_path.exists():
            LOGGER.warning(f"Exported ONNX model not found at {input_model_path}. Skipping conversion.")
            return CommandOutput(status=CommandStatus.SKIPPED)
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        custom_trt_profile = trt_profile
        trt_profile = self._get_trt_profile(
            dataloader_trt_profile=dataloader_trt_profile, custom_trt_profile=custom_trt_profile
        )

        with ExecutionContext(workspace=workspace):
            onnx_runner = OnnxrtCPURunner(
                model=input_model_path, input_metadata=input_metadata, output_metadata=output_metadata
            )
            with onnx_runner:
                onnx_input_metadata = onnx_runner.get_onnx_input_metadata()

        convert_cmd = ["polygraphy", "convert", input_model_path.relative_to(workspace).as_posix()]
        convert_cmd.extend(["--convert-to", "trt"])
        convert_cmd.extend(["-o", converted_model_temp_path.relative_to(workspace).as_posix()])

        if precision_mode == TensorRTPrecisionMode.HIERARCHY:
            trt_precision_flags = {
                TensorRTPrecision.FP32: ["--tf32"],
                TensorRTPrecision.FP16: ["--tf32", "--fp16"],
                TensorRTPrecision.INT8: ["--tf32", "--fp16", "--int8"],
            }[precision]
        elif precision_mode == TensorRTPrecisionMode.SINGLE:
            trt_precision_flags = {
                TensorRTPrecision.FP32: ["--tf32"],
                TensorRTPrecision.FP16: ["--fp16"],
                TensorRTPrecision.INT8: ["--int8"],
            }[precision]
        else:
            trt_precision_flags = None

        if trt_precision_flags:
            convert_cmd.extend(trt_precision_flags)

        if max_workspace_size is not None:
            if tensorrt.get_version() < LooseVersion("8.4.0"):
                convert_cmd.extend(["--workspace", f"{max_workspace_size}"])
            else:
                convert_cmd.extend(["--pool-limit", f"workspace:{max_workspace_size}"])

        def get_args(max_batch_size=None):
            return convert_cmd + self._get_shape_args(
                onnx_input_metadata=onnx_input_metadata,
                trt_profile=trt_profile,
                batch_dim=batch_dim,
                max_batch_size=max_batch_size,
            )

        with ExecutionContext(
            workspace=workspace,
            cmd_path=converted_model_path.parent / "reproduce_conversion.sh",
            verbose=verbose,
        ) as context:
            kwargs = {
                "batch_dim": batch_dim,
                "model_path": converted_model_temp_path,
                "runner_name": TensorRTRunner.name(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
            }

            load_args = parse_kwargs_to_cmd(kwargs)
            from model_navigator.commands.convert.onnx import trt_load_script

            def convert_func(args):
                try:
                    context.execute_cmd(args + ["&&", sys.executable, trt_load_script.__file__] + load_args)
                    converted_model_temp_path.replace(converted_model_path)
                finally:
                    converted_model_temp_path.unlink(missing_ok=True)

            self._execute_conversion(
                convert_func=convert_func,
                get_args=get_args,
                batch_dim=batch_dim,
                device_max_batch_size=device_max_batch_size,
                dataloader_max_batch_size=dataloader_max_batch_size,
                custom_trt_profile_available=bool(custom_trt_profile),
            )
        LOGGER.info("Converted ONNX to TensorRT.")
        return CommandOutput(status=CommandStatus.OK)

    @staticmethod
    def _get_shape_args(
        onnx_input_metadata: TensorMetadata,
        trt_profile: TensorRTProfile,
        batch_dim: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        if batch_dim is not None and max_batch_size is not None and max_batch_size > 0:
            trt_profile = tensorrt.get_trt_profile_with_new_max_batch_size(
                trt_profile=trt_profile,
                max_batch_size=max_batch_size,
                batch_dim=batch_dim,
            )

        shape_args = []
        for attr in ("min", "opt", "max"):
            arg = f"--trt-{attr}-shapes"
            shapes = []
            for input_name in trt_profile:
                if input_name not in onnx_input_metadata:
                    continue
                shape = ",".join([str(d) for d in getattr(trt_profile[input_name], attr)])
                shapes.append(f"{input_name}:[{shape}]")
            if shapes:
                shape_args.extend([f"{arg}"] + shapes)

        return shape_args
