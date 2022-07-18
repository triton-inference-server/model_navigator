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

import tensorflow as tf  # pytype: disable=import-error

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.convert.base import ConvertBase
from model_navigator.framework_api.commands.convert.converters import sm2tftrt
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.exceptions import ExecutionContext
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class ConvertSavedModel2ONNX(ConvertBase):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        # pytype: disable=wrong-arg-types
        super().__init__(
            name="Convert SavedModel to ONNX",
            command_type=CommandType.CONVERT,
            target_format=Format.ONNX,
            requires=requires,
        )
        # pytype: enable=wrong-arg-types

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format)

    def __call__(
        self,
        workdir: Path,
        opset: int,
        model_name: str,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        **kwargs,
    ):
        LOGGER.info("SavedModel to ONNX conversion started")
        exported_model_path = get_package_path(workdir, model_name) / ExportTF2SavedModel().get_output_relative_path()
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()

        convert_cmd = [
            "python",
            "-mtf2onnx.convert",
            "--saved-model",
            exported_model_path.as_posix(),
            "--output",
            converted_model_path.as_posix(),
            "--opset",
            str(opset),
            "--rename-inputs",
            ",".join(input_metadata.keys()),
            "--rename-outputs",
            ",".join(output_metadata.keys()),
        ]

        with ExecutionContext() as context:
            context.execute_cmd(convert_cmd)

        return self.get_output_relative_path()


class ConvertSavedModel2TFTRT(ConvertBase):
    def __init__(self, target_precision: TensorRTPrecision, requires: Tuple[Command, ...] = ()):
        # pytype: disable=wrong-arg-types
        super().__init__(
            name="Convert SavedModel to TF-TRT",
            command_type=CommandType.CONVERT,
            target_format=Format.TF_TRT,
            requires=requires,
        )
        self.target_precision = target_precision
        # pytype: enable=wrong-arg-types

    def _get_loggers(self) -> list:
        return [tf.get_logger()]

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format, precision=self.target_precision)

    def __call__(
        self,
        max_workspace_size: int,
        minimum_segment_size: int,
        workdir: Path,
        model_name: str,
        batch_dim: Optional[int] = None,
        **kwargs,
    ) -> Optional[Path]:
        # for precision in target_precisions:

        # generate samples as tuples for TF-TRT converter

        exported_model_path = get_package_path(workdir, model_name) / ExportTF2SavedModel().get_output_relative_path()
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        converted_model_path.parent.mkdir(parents=True, exist_ok=True)

        with ExecutionContext(converted_model_path.parent / "reproduce.py") as context:
            kwargs = {
                "exported_model_path": exported_model_path.as_posix(),
                "converted_model_path": converted_model_path.as_posix(),
                "max_workspace_size": max_workspace_size,
                "target_precision": self.target_precision.value,
                "minimum_segment_size": minimum_segment_size,
                "package_path": get_package_path(workdir, model_name).as_posix(),
                "batch_dim": batch_dim,
            }

            args = []
            for k, v in kwargs.items():
                s = str(v).replace("'", '"')
                args.extend([f"--{k}", s])

            context.execute_external_runtime_script(sm2tftrt.__file__, args)

        return self.get_output_relative_path()
