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

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch  # pytype: disable=import-error

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export.pyt import ExportPYT2TorchScript
from model_navigator.framework_api.common import Format, TensorMetadata
from model_navigator.framework_api.exceptions import UserErrorContext
from model_navigator.framework_api.utils import (
    Framework,
    format_to_relative_model_path,
    get_base_format,
    get_package_path,
    numpy_to_torch_dtype,
)


class ConvertTorchScript2TorchTensorRT(Command):
    def __init__(self, target_format: Format, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Convert TorschScript to TorchTensorRT",
            command_type=CommandType.CONVERT,
            target_format=target_format,
            requires=requires,
        )

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format)

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        input_metadata: TensorMetadata,
        trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
        **kwargs,
    ) -> Optional[Path]:
        import torch_tensorrt  # pytype: disable=import-error

        ts_format = get_base_format(self.target_format, Framework.PYT)
        exported_model_path = (
            get_package_path(workdir, model_name)
            / ExportPYT2TorchScript(target_format=ts_format).get_output_relative_path()
        )
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if converted_model_path.is_file() or converted_model_path.is_dir():
            return None
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        trt_casts = {np.dtype(np.int64): np.int32}
        input_dtypes = [
            numpy_to_torch_dtype(trt_casts.get(input_spec.dtype, input_spec.dtype))
            for input_spec in input_metadata.values()
        ]

        all_shapes = {}
        for input_name, spec in input_metadata.items():
            shapes = {}
            for i, shape_type in enumerate(("min", "opt", "max")):
                tensor_shape = list(spec.shape)
                for ax, val in trt_dynamic_axes[input_name].items():
                    tensor_shape[ax] = val[i]
                shapes[shape_type] = tensor_shape
            all_shapes[input_name] = shapes

        model_input_shapes = []
        for input_shapes, input_dtype in zip(all_shapes.values(), input_dtypes):
            model_input_shapes.append(
                torch_tensorrt.Input(
                    min_shape=input_shapes["min"],
                    opt_shape=input_shapes["opt"],
                    max_shape=input_shapes["max"],
                    dtype=input_dtype,
                )
            )

        model = torch.jit.load(exported_model_path.as_posix())
        with UserErrorContext():
            tr_model_compiled = torch_tensorrt.compile(model, inputs=model_input_shapes)
            tr_model_compiled.save(converted_model_path.as_posix())

        return self.get_output_relative_path()
