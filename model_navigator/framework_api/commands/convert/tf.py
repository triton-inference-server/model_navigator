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
from typing import List, Optional, Tuple

from tensorflow.python.compiler.tensorrt import trt_convert as trtc  # pytype: disable=import-error

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.common import Sample
from model_navigator.framework_api.errors import ExternalErrorContext
from model_navigator.framework_api.utils import (
    ArtifactType,
    format_to_relative_model_path,
    get_package_path,
    sample_to_tuple,
)
from model_navigator.model import Format


class ConvertSavedModel2TFTRT(Command):
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
        results = {}
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
        with ExternalErrorContext():
            converter = trtc.TrtGraphConverterV2(
                input_saved_model_dir=exported_model_path.as_posix(), use_dynamic_shape=True, conversion_params=params
            )

            converter.convert()
            converter.build(_dataloader)
            converter.save(converted_model_path.as_posix())

        results[f"{ArtifactType.CONVERTED_MODEL_PATH.value}_{self.target_precision.value}"] = converted_model_path

        return self.get_output_relative_path()
