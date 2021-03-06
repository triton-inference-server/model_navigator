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
import dataclasses
import pathlib
from typing import Optional

from model_navigator.common.config import BatchingConfig, TensorRTCommonConfig
from model_navigator.converter.config import DatasetProfileConfig
from model_navigator.model import Model, ModelSignatureConfig
from model_navigator.perf_analyzer import PerfMeasurementConfig
from model_navigator.results import Status
from model_navigator.triton import (
    RunTritonConfig,
    TritonBatchingConfig,
    TritonCustomBackendParametersConfig,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
)


@dataclasses.dataclass
class TritonConfiguratorResult:
    status: Status
    model: Model
    model_config_name: str
    model_config_path: Optional[pathlib.Path] = None
    batching_config: Optional[BatchingConfig] = None
    triton_batching_config: Optional[TritonBatchingConfig] = None
    backend_config: Optional[TritonCustomBackendParametersConfig] = None
    instances_config: Optional[TritonModelInstancesConfig] = None
    tensorrt_common_config: Optional[TensorRTCommonConfig] = None
    dataset_profile_config: Optional[DatasetProfileConfig] = None
    perf_measurement_config: Optional[PerfMeasurementConfig] = None
    optimization_config: Optional[TritonModelOptimizationConfig] = None
    triton_config: Optional[RunTritonConfig] = None
    model_signature_config: Optional[ModelSignatureConfig] = None
