# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class ConvertONNX2TRT(Command):
    trt_precision_to_arg = {
        TensorRTPrecision.FP32: "",
        TensorRTPrecision.TF32: "--tf32",
        TensorRTPrecision.FP16: "--fp16",
        TensorRTPrecision.INT8: "--int8",
    }

    def __init__(self, target_precision: TensorRTPrecision):
        # pytype: disable=wrong-arg-types
        super().__init__(
            name="Convert ONNX to TensorRT",
            command_type=CommandType.CONVERT,
            target_format=Format.TENSORRT,
        )
        self.target_precision = target_precision
        # pytype: enable=wrong-arg-types

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format, precision=self.target_precision)

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        samples: List,
        input_metadata: TensorMetadata,
        max_workspace_size: Optional[int] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
        **kwargs,
    ) -> Optional[Path]:

        exported_model_path = get_package_path(workdir, model_name) / ExportPYT2ONNX().get_output_relative_path()
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()

        if converted_model_path.is_file() or converted_model_path.is_dir():
            return None
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        convert_cmd = ["polygraphy", "convert", exported_model_path.as_posix()]
        convert_cmd.extend(["--convert-to", "trt"])
        convert_cmd.extend(["-o", converted_model_path.as_posix()])

        if dynamic_axes is not None:
            if trt_dynamic_axes is None:

                trt_dynamic_axes_shapes = {
                    name: {ax: [] for ax in axes} for name, axes in dynamic_axes.items() if name in input_metadata
                }
                trt_dynamic_axes = {}
                for sample in samples:
                    for name, tensor in sample.items():
                        for i, dim in enumerate(tensor.shape):
                            trt_dynamic_axes_shapes[name][i].append(dim)

                for name, axes in trt_dynamic_axes_shapes.items():
                    trt_dynamic_axes[name] = {}
                    for ax, shapes in axes.items():
                        trt_dynamic_axes[name][ax] = (min(shapes), int(np.median(shapes)), max(shapes))

                LOGGER.warning(
                    f"No TRT (min, opt, max) values for axes provided. Using values derived from samples: {trt_dynamic_axes}"
                )

            sample = samples[0]
            for i, arg in enumerate(("--trt-min-shapes", "--trt-opt-shapes", "--trt-max-shapes")):
                shapes = []
                for input_name, tensor in sample.items():
                    tensor_shape = list(tensor.shape)
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

        subprocess.run(convert_cmd, check=True)

        return self.get_output_relative_path()
