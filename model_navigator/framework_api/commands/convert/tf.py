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
from typing import List, Optional, Tuple

from tensorflow.python.compiler.tensorrt import trt_convert as trtc  # pytype: disable=import-error

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.convert.base import ConvertBase
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.common import Sample, TensorMetadata
from model_navigator.framework_api.exceptions import UserErrorContext
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import format_to_relative_model_path, get_package_path, sample_to_tuple
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

        with UserErrorContext():
            output = subprocess.run(convert_cmd, check=True, capture_output=True)
            self._log_subprocess_output(output=output)

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

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format, precision=self.target_precision)

    def __call__(
        self,
        max_workspace_size: int,
        minimum_segment_size: int,
        workdir: Path,
        model_name: str,
        conversion_samples: List[Sample],
        **kwargs,
    ) -> Optional[Path]:
        # for precision in target_precisions:

        # generate samples as tuples for TF-TRT converter
        def _dataloader():
            for sample in conversion_samples:
                yield sample_to_tuple(sample)

        exported_model_path = get_package_path(workdir, model_name) / ExportTF2SavedModel().get_output_relative_path()
        converted_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()

        params = trtc.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            max_workspace_size_bytes=max_workspace_size,
            precision_mode=self.target_precision.value,
            minimum_segment_size=minimum_segment_size,
        )
        # TODO: allow setting dynamic_shape_profile_strategy
        with UserErrorContext():
            converter = trtc.TrtGraphConverterV2(
                input_saved_model_dir=exported_model_path.as_posix(), use_dynamic_shape=True, conversion_params=params
            )

            converter.convert()
            converter.build(_dataloader)
            converter.save(converted_model_path.as_posix())

        return self.get_output_relative_path()
