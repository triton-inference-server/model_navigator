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
import abc
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from model_navigator.triton import DeviceKind
from model_navigator.utils import Workspace
from model_navigator.utils.config import BaseConfig


class TritonLaunchMode(Enum):
    LOCAL = "local"
    DOCKER = "docker"


class BaseConfigGenerator(abc.ABC):
    def __init__(self, *, workspace: Workspace, verbose: int = 0):
        self._workspace = workspace
        self._verbose = verbose
        analyzer_path = workspace.path / "analyzer"
        self._analyzer_path = analyzer_path
        self._analyzer_checkpoints_dir_path = self._analyzer_path / "checkpoints"
        self._output_model_repository_path = self._analyzer_path / "model-store"

    @property
    def analyzer_path(self) -> pathlib.Path:
        return self._analyzer_path

    @property
    def output_model_repository_path(self) -> pathlib.Path:
        return self._output_model_repository_path.resolve()

    @abc.abstractmethod
    def generate_config(self, **kwargs):
        pass


@dataclass
class ModelAnalyzerTritonConfig(BaseConfig):
    model_repository: pathlib.Path = pathlib.Path("model-store")
    triton_launch_mode: TritonLaunchMode = TritonLaunchMode.LOCAL
    triton_server_path: str = "tritonserver"


@dataclass
class ModelAnalyzerProfileConfig(BaseConfig):
    max_concurrency: int = 1024
    max_instance_count: int = 5
    max_batch_size: int = 32
    concurrency: Optional[List[int]] = None
    instance_counts: Optional[Dict[DeviceKind, List]] = None
    preferred_batch_sizes: Optional[List[int]] = None


@dataclass
class ModelAnalyzerAnalysisConfig(BaseConfig):
    top_n_configs: int = 3
    objectives: Dict[str, int] = field(default_factory=lambda: {"perf_throughput": 10})
    max_latency_ms: Optional[int] = None
    min_throughput: int = 1
    max_gpu_usage_mb: Optional[int] = None
