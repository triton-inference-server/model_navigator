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
from typing import Optional, Tuple

import numpy as np
from polygraphy.backend.trt import Profile

from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api.commands.convert.base import ConvertBase
from model_navigator.framework_api.commands.convert.converters import ts2torchtrt
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export.pyt import ExportPYT2TorchScript
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.exceptions import ExecutionContext
from model_navigator.framework_api.logger import LOGGER, get_pytorch_loggers_names
from model_navigator.framework_api.utils import JitType, format_to_relative_model_path, get_package_path
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
        trt_profile: Profile,
        **kwargs,
    ) -> Optional[Path]:
        LOGGER.info("Conversion TorchScript to TorchTensorRT started")

        import torch_tensorrt  # pytype: disable=import-error # noqa: F401

        exported_model_path = (
            get_package_path(workdir, model_name)
            / ExportPYT2TorchScript(target_jit_type=self.target_jit_type).get_output_relative_path()
        )
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if converted_model_path.is_file() or converted_model_path.is_dir():
            return self.get_output_relative_path()
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        trt_casts = {np.dtype(np.int64): np.dtype(np.int32)}
        input_dtypes_str = [
            trt_casts.get(input_spec.dtype, input_spec.dtype).name for input_spec in input_metadata.values()
        ]

        with ExecutionContext(converted_model_path.parent / "reproduce.py") as context:
            kwargs = {
                "exported_model_path": exported_model_path.as_posix(),
                "converted_model_path": converted_model_path.as_posix(),
                "shapes": {name: vars(shape_tuple) for name, shape_tuple in trt_profile.items()},
                "input_dtypes": input_dtypes_str,
                "max_workspace_size": max_workspace_size,
                "precision": self.target_precision.value,
                "precision_mode": precision_mode.value,
            }

            args = []
            for k, v in kwargs.items():
                s = str(v).replace("'", '"')
                args.extend([f"--{k}", s])

            context.execute_external_runtime_script(ts2torchtrt.__file__, args)

        return self.get_output_relative_path()
