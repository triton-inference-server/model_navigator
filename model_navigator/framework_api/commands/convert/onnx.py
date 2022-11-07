# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from distutils.version import LooseVersion
from pathlib import Path
from typing import Optional, Tuple

from polygraphy.backend.onnxrt import SessionFromOnnx
from polygraphy.backend.trt import Profile

from model_navigator.framework_api.exceptions import UserError
from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api.commands.convert.base import ConvertBase
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.execution_context import ExecutionContext
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.runners.onnx import OnnxrtRunner
from model_navigator.framework_api.utils import Framework, Status, format_to_relative_model_path
from model_navigator.model import Format
from model_navigator.utils import devices, tensorrt


class ConvertONNX2TRT(ConvertBase):
    def __init__(
        self,
        target_precision: TensorRTPrecision,
        precision_mode: TensorRTPrecisionMode,
        enable_xla: Optional[bool] = None,
        jit_compile: Optional[bool] = None,
        requires: Tuple[Command, ...] = (),
    ):
        # pytype: disable=wrong-arg-types
        super().__init__(
            name="Convert ONNX to TensorRT",
            command_type=CommandType.CONVERT,
            target_format=Format.TENSORRT,
            requires=requires,
        )
        self.precision_mode = precision_mode
        self.target_precision = target_precision
        self.enable_xla = enable_xla
        self.jit_compile = jit_compile
        # pytype: enable=wrong-arg-types

    def __call__(
        self,
        workdir: Path,
        framework: Framework,
        model_name: str,
        target_device: str,
        trt_profile: Profile,
        verbose: bool,
        max_workspace_size: Optional[int] = None,
        **kwargs,
    ) -> Optional[Path]:
        LOGGER.info("ONNX to TRT conversion started")
        if not devices.get_available_gpus():
            raise RuntimeError("No GPUs available.")

        if framework == Framework.PYT:
            from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX

            input_model_path = workdir / ExportPYT2ONNX().get_output_relative_path()
        elif framework in (Framework.TF2, Framework.JAX):
            input_model_path = workdir / format_to_relative_model_path(
                format=Format.ONNX,
                enable_xla=self.enable_xla,
                jit_compile=self.jit_compile,
            )
        elif framework == Framework.ONNX:  # ONNX
            from model_navigator.framework_api.commands.copy.onnx import CopyONNX

            input_model_path = workdir / CopyONNX().get_output_relative_path()
        else:
            raise UserError(f"Unknown framework: {framework.value}")
        converted_model_path = workdir / self.get_output_relative_path()

        if converted_model_path.exists():
            LOGGER.info("Model already exists. Skipping conversion.")
            return self.get_output_relative_path()
        if not input_model_path.exists():
            LOGGER.warning(f"Exported ONNX model not found at {input_model_path}. Skipping conversion.")
            self.status = Status.SKIPPED
            return
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        with ExecutionContext(workdir=workdir):
            onnx_runner = OnnxrtRunner(SessionFromOnnx(input_model_path.as_posix(), providers=[target_device]))
            with onnx_runner:
                onnx_input_metadata = onnx_runner.get_input_metadata()

        convert_cmd = ["polygraphy", "convert", input_model_path.relative_to(workdir).as_posix()]
        convert_cmd.extend(["--convert-to", "trt"])
        convert_cmd.extend(["-o", converted_model_path.relative_to(workdir).as_posix()])

        if self.precision_mode == TensorRTPrecisionMode.HIERARCHY:
            trt_precision_flags = {
                TensorRTPrecision.FP32: ["--tf32"],
                TensorRTPrecision.FP16: ["--tf32", "--fp16"],
                TensorRTPrecision.INT8: ["--tf32", "--fp16", "--int8"],
            }[self.target_precision]
        elif self.precision_mode == TensorRTPrecisionMode.SINGLE:
            trt_precision_flags = {
                TensorRTPrecision.FP32: ["--tf32"],
                TensorRTPrecision.FP16: ["--fp16"],
                TensorRTPrecision.INT8: ["--int8"],
            }[self.target_precision]
        else:
            trt_precision_flags = None

        if trt_precision_flags:
            convert_cmd.extend(trt_precision_flags)

        # for i, arg in enumerate(("--trt-min-shapes", "--trt-opt-shapes", "--trt-max-shapes")):
        for attr in ("min", "opt", "max"):
            arg = f"--trt-{attr}-shapes"
            shapes = []
            for input_name in trt_profile:
                if input_name not in onnx_input_metadata:
                    continue
                shape = ",".join([str(d) for d in getattr(trt_profile[input_name], attr)])
                shapes.append(f"{input_name}:[{shape}]")
            if shapes:
                convert_cmd.extend([f"{arg}"] + shapes)

        if max_workspace_size is not None:
            if tensorrt.version() < LooseVersion("8.4.0"):
                convert_cmd.extend(["--workspace", f"{max_workspace_size}"])
            else:
                convert_cmd.extend(["--pool-limit", f"workspace:{max_workspace_size}"])

        with ExecutionContext(
            workdir=workdir,
            cmd_path=converted_model_path.parent / "reproduce_conversion.sh",
            verbose=verbose,
        ) as context:
            context.execute_cmd(convert_cmd)

        return self.get_output_relative_path()
