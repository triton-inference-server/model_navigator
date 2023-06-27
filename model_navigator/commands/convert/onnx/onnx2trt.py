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
import json
import pathlib
import sys
import tempfile
from distutils.version import LooseVersion
from typing import Optional

from model_navigator.api.config import (
    TensorRTCompatibilityLevel,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TensorRTProfile,
)
from model_navigator.commands.base import CommandOutput, CommandStatus
from model_navigator.commands.convert.base import Convert2TensorRTWithMaxBatchSizeSearch
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.runners.tensorrt import TensorRTRunner
from model_navigator.utils import devices
from model_navigator.utils.common import parse_kwargs_to_cmd


class ConvertONNX2TRT(Convert2TensorRTWithMaxBatchSizeSearch):
    """Command that converts ONNX checkpoint to TensorRT model plan."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        parent_path: pathlib.Path,
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
        optimization_level: Optional[int] = None,
        compatibility_level: Optional[TensorRTCompatibilityLevel] = None,
        verbose: bool = False,
    ) -> CommandOutput:
        """Run the ConvertONNX2TRT Command.

        Args:
            workspace: Model Navigator working directory.
            path: ONNX checkpoint path, relative to workspace.
            parent_path: Path of ONNX parent model, relative to workspace.
            input_metadata: Model input metadata.
            output_metadata: Model output metadata.
            precision: TensorRT precision.
            precision_mode: TensorRT precision mode.
            dataloader_trt_profile: Dataloader TensorRT profile.
            max_workspace_size: Maximum TensorRT workspace size, in bytes. Defaults to None.
            batch_dim: Dimension of the batching, None if model does not support batching.
                Defaults to None.
            dataloader_max_batch_size (Optional[int], optional): Maximum batch size in the dataloader. Defaults to None.
            device_max_batch_size: Maximum batch size that fits on the device.
                Defaults to None.
            trt_profile: User specified TensorRT profile. Defaults to None.
            optimization_level: Optimization level for TensorRT engine
            compatibility_level: Hardware compatibility level for generated engine
            verbose: enable verbose logging for command

        Returns:
            CommandOutput: Status and results of the command.
        """
        LOGGER.info("ONNX to TRT conversion started")
        if not devices.get_available_gpus():
            raise RuntimeError("No GPUs available.")

        input_model_path = workspace.path / parent_path
        converted_model_path = workspace.path / path

        if not input_model_path.exists():
            LOGGER.warning(f"Exported ONNX model not found at {input_model_path}. Skipping conversion.")
            return CommandOutput(status=CommandStatus.SKIPPED)
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        custom_trt_profile = trt_profile
        trt_profile = self._get_trt_profile(
            dataloader_trt_profile=dataloader_trt_profile, custom_trt_profile=custom_trt_profile
        )

        onnx_input_metadata = self._get_onnx_input_metadata(
            input_model_path=input_model_path,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            workspace=workspace,
            reproduce_script_path=converted_model_path.parent,
            verbose=verbose,
        )

        convert_cmd = ["polygraphy", "convert", input_model_path.relative_to(workspace.path).as_posix()]
        convert_cmd.extend(["--convert-to", "trt"])
        convert_cmd.extend(["-o", converted_model_path.relative_to(workspace.path).as_posix()])

        if optimization_level is not None:
            convert_cmd.extend(["--builder-optimization-level", optimization_level])

        if compatibility_level is not None:
            convert_cmd.extend(["--hardware-compatibility-level", compatibility_level.value])

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
            if tensorrt_utils.get_version() < LooseVersion("8.4.0"):
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
                "model_path": converted_model_path.relative_to(workspace.path).as_posix(),
                "runner_name": TensorRTRunner.name(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
            }

            load_args = parse_kwargs_to_cmd(kwargs)
            from . import trt_load_script

            max_conversion_batch_size = self._execute_conversion(
                convert_func=lambda args: context.execute_cmd(
                    args + ["&&", sys.executable, trt_load_script.__file__] + load_args
                ),
                get_args=get_args,
                batch_dim=batch_dim,
                device_max_batch_size=device_max_batch_size,
                dataloader_max_batch_size=dataloader_max_batch_size,
                custom_trt_profile_available=bool(custom_trt_profile),
            )

        LOGGER.info("Converted ONNX to TensorRT.")
        return CommandOutput(status=CommandStatus.OK, output={"max_conversion_batch_size": max_conversion_batch_size})

    @staticmethod
    def _get_shape_args(
        onnx_input_metadata: TensorMetadata,
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

    def _get_onnx_input_metadata(
        self,
        workspace: Workspace,
        input_model_path: pathlib.Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        reproduce_script_path: pathlib.Path,
        verbose: bool,
    ):
        with ExecutionContext(
            script_path=reproduce_script_path / "reproduce_onnx_input_metadata.py",
            cmd_path=reproduce_script_path / "reproduce_onnx_input_metadata.sh",
            workspace=workspace,
            verbose=verbose,
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "model_path": input_model_path.relative_to(workspace.path).as_posix(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
                "results_path": temp_file.name,
            }
            args = parse_kwargs_to_cmd(kwargs)
            from . import collect_onnx_input_metadata

            try:
                context.execute_external_runtime_script(collect_onnx_input_metadata.__file__, args)
                with open(temp_file.name) as fp:
                    input_metadata = json.load(fp)
                LOGGER.info("Input metadata collected from ONNX model.")
            except Exception as e:
                LOGGER.warning(
                    "Unable to collect metadata from ONNX model. The evaluation failed. Empty metadata used."
                )
                LOGGER.warning(f"Error during obtaining metadata: {str(e)}")
                input_metadata = []

            return TensorMetadata.from_json(input_metadata)
