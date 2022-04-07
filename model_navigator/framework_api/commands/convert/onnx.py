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

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from polygraphy.backend.onnxrt import SessionFromOnnx

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.exceptions import UserError, UserErrorContext
from model_navigator.framework_api.runners.onnx import OnnxrtRunner
from model_navigator.framework_api.utils import Framework, format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class ConvertONNX2TRT(Command):
    trt_precision_to_arg = {
        TensorRTPrecision.FP32: "",
        TensorRTPrecision.TF32: "--tf32",
        TensorRTPrecision.FP16: "--fp16",
        TensorRTPrecision.INT8: "--int8",
    }

    def __init__(self, target_precision: TensorRTPrecision, requires: Tuple[Command, ...] = ()):
        # pytype: disable=wrong-arg-types
        super().__init__(
            name="Convert ONNX to TensorRT",
            command_type=CommandType.CONVERT,
            target_format=Format.TENSORRT,
            requires=requires,
        )
        self.target_precision = target_precision
        # pytype: enable=wrong-arg-types

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format, precision=self.target_precision)

    def __call__(
        self,
        workdir: Path,
        framework: Framework,
        model_name: str,
        input_metadata: TensorMetadata,
        target_device: str,
        model: Optional[Path] = None,
        max_workspace_size: Optional[int] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
        **kwargs,
    ) -> Optional[Path]:

        if framework == Framework.PYT:
            from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX

            input_model_path = (
                get_package_path(workdir, model_name) / ExportPYT2ONNX().get_output_relative_path()
            ).as_posix()
        elif framework == Framework.TF2:
            from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2ONNX

            input_model_path = (
                get_package_path(workdir, model_name) / ConvertSavedModel2ONNX().get_output_relative_path()
            ).as_posix()
        elif framework == Framework.ONNX:  # ONNX
            # pytype: disable=attribute-error
            input_model_path = model.as_posix()
            # pytype: enable=attribute-error
        else:
            raise UserError(f"Unknown framework: {framework.value}")
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()

        if converted_model_path.is_file() or converted_model_path.is_dir():
            return None
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        with UserErrorContext():
            onnx_runner = OnnxrtRunner(SessionFromOnnx(input_model_path, providers=[target_device]))
            with onnx_runner:
                onnx_input_metadata = onnx_runner.get_input_metadata()

        convert_cmd = ["polygraphy", "convert", input_model_path]
        convert_cmd.extend(["--convert-to", "trt"])
        convert_cmd.extend(["-o", converted_model_path.as_posix()])

        if dynamic_axes is not None:
            for i, arg in enumerate(("--trt-min-shapes", "--trt-opt-shapes", "--trt-max-shapes")):
                shapes = []
                for input_name, spec in input_metadata.items():
                    if input_name not in onnx_input_metadata:
                        continue
                    tensor_shape = list(spec.shape)
                    for ax, val in trt_dynamic_axes[input_name].items():
                        tensor_shape[ax] = val[i]
                    shape = ",".join([str(d) for d in tensor_shape])
                    shapes.append(f"{input_name}:[{shape}]")
                convert_cmd.extend([f"{arg}"] + shapes)

        precision_arg = self.trt_precision_to_arg[self.target_precision]
        if precision_arg:
            convert_cmd.append(precision_arg)

        if max_workspace_size is not None:
            convert_cmd.append(f"--workspace={max_workspace_size}")

        with UserErrorContext():
            subprocess.run(convert_cmd, check=True)

        return self.get_output_relative_path()
