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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.converter import ComparatorConfig, ConversionConfig, DatasetProfileConfig
from model_navigator.model import ModelConfig, ModelSignatureConfig
from model_navigator.results import Status
from model_navigator.triton import TritonModelInstancesConfig, TritonModelOptimizationConfig, TritonModelSchedulerConfig


@dataclass
class HelmChartGenerationResult:
    status: Status
    triton_docker_image: str
    framework_docker_image: str
    src_model_config: ModelConfig
    src_model_signature_config: ModelSignatureConfig
    conversion_config: ConversionConfig
    tensorrt_common_config: TensorRTCommonConfig
    comparator_config: ComparatorConfig
    dataset_profile_config: DatasetProfileConfig
    optimization_config: TritonModelOptimizationConfig
    scheduler_config: TritonModelSchedulerConfig
    instances_config: TritonModelInstancesConfig
    helm_chart_dir_path: Optional[Path]
