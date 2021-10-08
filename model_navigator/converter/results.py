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
import dataclasses
from typing import Optional

from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.converter.config import ComparatorConfig, ConversionConfig, DatasetProfileConfig
from model_navigator.model import Model, ModelConfig, ModelSignatureConfig
from model_navigator.results import Status


@dataclasses.dataclass
class ConversionResult:
    status: Status
    source_model_config: ModelConfig
    conversion_config: Optional[ConversionConfig] = None
    tensorrt_common_config: Optional[TensorRTCommonConfig] = None
    model_signature_config: Optional[ModelSignatureConfig] = None
    comparator_config: Optional[ComparatorConfig] = None
    dataset_profile: Optional[DatasetProfileConfig] = None
    output_model: Optional[Model] = None
    framework_docker_image: Optional[str] = None
