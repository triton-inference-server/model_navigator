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
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import JitType, Status, parse_kwargs_to_cmd
from model_navigator.model import Format
from model_navigator.utils import devices


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

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        input_metadata: TensorMetadata,
        precision_mode: TensorRTPrecisionMode,
        max_workspace_size: int,
        trt_profile: Profile,
        target_device: str,
        **kwargs,
    ) -> Optional[Path]:
        LOGGER.info("Conversion TorchScript to TorchTensorRT started")
        if not devices.get_available_gpus():
            raise RuntimeError("No GPUs available.")

        import torch_tensorrt  # pytype: disable=import-error # noqa: F401

        exported_model_path = (
            workdir / ExportPYT2TorchScript(target_jit_type=self.target_jit_type).get_output_relative_path()
        )
        converted_model_path = workdir / self.get_output_relative_path()
        if converted_model_path.exists():
            LOGGER.info("Model already exists. Skipping conversion.")
            return self.get_output_relative_path()
        if not exported_model_path.exists():
            LOGGER.warning(f"Exported TorchScript model not found at {exported_model_path}. Skipping conversion.")
            self.status = Status.SKIPPED
            return

        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        trt_casts = {np.dtype(np.int64): np.dtype(np.int32)}
        input_dtypes_str = [
            trt_casts.get(input_spec.dtype, input_spec.dtype).name for input_spec in input_metadata.values()
        ]

        with ExecutionContext(
            workdir=workdir,
            script_path=converted_model_path.parent / "reproduce_conversion.py",
            cmd_path=converted_model_path.parent / "reproduce_conversion.sh",
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workdir).as_posix(),
                "converted_model_path": converted_model_path.relative_to(workdir).as_posix(),
                "shapes": {name: vars(shape_tuple) for name, shape_tuple in trt_profile.items()},
                "input_dtypes": input_dtypes_str,
                "max_workspace_size": max_workspace_size,
                "precision": self.target_precision.value,
                "precision_mode": precision_mode.value,
                "target_device": target_device,
                "navigator_workdir": workdir.as_posix(),
            }

            args = parse_kwargs_to_cmd(kwargs, (list, dict, tuple))

            context.execute_external_runtime_script(ts2torchtrt.__file__, args)

        return self.get_output_relative_path()
