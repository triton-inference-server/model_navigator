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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from model_navigator.converter import DatasetProfileConfig
from model_navigator.model_analyzer.config import (
    ModelAnalyzerAnalysisConfig,
    ModelAnalyzerProfileConfig,
    ModelAnalyzerTritonConfig,
)
from model_navigator.results import Status
from model_navigator.triton import (
    TritonDynamicBatchingConfig,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
)
from model_navigator.triton.config import TritonBatchingConfig


@dataclass
class ProfileResult:
    status: Status
    triton_docker_image: str
    profile_config: ModelAnalyzerProfileConfig
    triton_config: ModelAnalyzerTritonConfig
    dataset_profile: Optional[DatasetProfileConfig] = None
    profiling_data_path: Optional[Path] = None
    profiling_results_path: Optional[Path] = None


@dataclass
class AnalyzeResult:
    status: Status
    model_repository: Path
    analysis_config: ModelAnalyzerAnalysisConfig
    model_name: Optional[str] = None
    model_config_path: Optional[str] = None
    optimization_config: Optional[TritonModelOptimizationConfig] = None
    batching_config: Optional[TritonBatchingConfig] = None
    dynamic_batching_config: Optional[TritonDynamicBatchingConfig] = None
    instances_config: Optional[TritonModelInstancesConfig] = None
    results_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
