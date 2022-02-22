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


from pathlib import Path
from typing import Optional

import tensorflow  # pytype: disable=import-error

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import CommandType
from model_navigator.framework_api.commands.performance.base import PerformanceBase
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.runners.tf import TFRunner, TFTRTRunner
from model_navigator.framework_api.utils import format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class PerformanceSavedModel(PerformanceBase):
    def __init__(self, target_format: Format, target_precision: Optional[TensorRTPrecision] = None):
        super().__init__(
            name="Performance SavedModel",
            command_type=CommandType.PERFORMANCE,
            target_format=target_format,
        )
        self.target_precision = target_precision

    def _get_runner(
        self,
        workdir: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        model_name: str,
        **kwargs,
    ):

        output_names = list(output_metadata.keys())
        exported_model_path = get_package_path(workdir, model_name) / format_to_relative_model_path(
            format=self.target_format,
            precision=self.target_precision,
        )

        if self.target_format == Format.TF_TRT:
            savedmodel_runner = TFTRTRunner(
                tensorflow.keras.models.load_model(exported_model_path),
                input_metadata=input_metadata,
                output_names=output_names,
            )
        else:
            savedmodel_runner = TFRunner(
                tensorflow.keras.models.load_model(exported_model_path),
                input_metadata=input_metadata,
                output_names=output_names,
            )

        return savedmodel_runner
