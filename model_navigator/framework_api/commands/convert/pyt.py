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

from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api.commands.convert.base import ConvertBase
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export.pyt import ExportPYT2TorchScript
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.exceptions import UserError, UserErrorContext
from model_navigator.framework_api.logger import LOGGER, get_pytorch_loggers_names
from model_navigator.framework_api.utils import (
    JitType,
    format_to_relative_model_path,
    get_package_path,
    numpy_to_torch_dtype,
)
from model_navigator.model import Format


class ConvertTorchScript2TorchTensorRT(ConvertBase):
    def __init__(
        self, target_jit_type: JitType, target_precision: TensorRTPrecision, requires: Tuple[Command, ...] = ()
    ):
        super().__init__(
            name="Convert TorschScript to TorchTensorRT",
            command_type=CommandType.CONVERT,
            target_format=Format.TORCH_TRT,
            requires=requires,
        )
        self.target_jit_type = target_jit_type
        self.target_precision = target_precision

    @staticmethod
    def _get_precision(precision, precision_mode):
        if precision_mode == TensorRTPrecisionMode.HIERARCHY:
            enabled_precisions = {
                TensorRTPrecision.FP32: [torch.float],
                TensorRTPrecision.TF32: [torch.float],
                TensorRTPrecision.FP16: [torch.float, torch.half],
                TensorRTPrecision.INT8: [torch.float, torch.half, torch.int8],
            }[precision]
        elif precision_mode == TensorRTPrecisionMode.SINGLE:
            enabled_precisions = {
                TensorRTPrecision.FP32: [torch.float],
                TensorRTPrecision.TF32: [torch.float],
                TensorRTPrecision.FP16: [torch.half],
                TensorRTPrecision.INT8: [torch.int8],
            }[precision]
        else:
            raise UserError(
                f"Unsupported precision mode {precision_mode}. Only {TensorRTPrecisionMode.HIERARCHY} and "
                "{TensorRTPrecisionMode.SINGLE} are allowed"
            )
        return {
            "enabled_precisions": enabled_precisions,
            "disable_tf32": precision == TensorRTPrecision.FP32,
        }

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(
            self.target_format, jit_type=self.target_jit_type, precision=self.target_precision
        )

    def _get_loggers(self) -> list:
        return get_pytorch_loggers_names()

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        input_metadata: TensorMetadata,
        precision_mode: TensorRTPrecisionMode,
        max_workspace_size: int,
        trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
        **kwargs,
    ) -> Optional[Path]:
        LOGGER.info("Conversion TorchScript to TorchTensorRT started")
        import torch_tensorrt  # pytype: disable=import-error

        exported_model_path = (
            get_package_path(workdir, model_name)
            / ExportPYT2TorchScript(target_jit_type=self.target_jit_type).get_output_relative_path()
        )
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if converted_model_path.is_file() or converted_model_path.is_dir():
            return self.get_output_relative_path()
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
            tr_model_compiled = torch_tensorrt.compile(
                module=model,
                inputs=model_input_shapes,
                workspace_size=max_workspace_size,
                truncate_long_and_double=True,
                **self._get_precision(self.target_precision, precision_mode),
            )
            tr_model_compiled.save(converted_model_path.as_posix())

        return self.get_output_relative_path()
