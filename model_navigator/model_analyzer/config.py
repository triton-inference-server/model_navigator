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
import abc
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from model_navigator.triton import DeviceKind
from model_navigator.triton.config import TritonLaunchMode
from model_navigator.utils import Workspace
from model_navigator.utils.config import BaseConfig


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

    @property
    def checkpoints_dir_path(self) -> pathlib.Path:
        return self._analyzer_checkpoints_dir_path

    @abc.abstractmethod
    def generate_config(self, **kwargs):
        pass


@dataclass
class ModelAnalyzerTritonConfig(BaseConfig):
    model_repository: pathlib.Path
    triton_launch_mode: TritonLaunchMode = TritonLaunchMode.LOCAL
    triton_server_path: str = "tritonserver"


@dataclass
class ModelAnalyzerProfileConfig(BaseConfig):
    config_search_max_batch_size: int = 128
    config_search_max_concurrency: int = 1024
    config_search_max_instance_count: int = 5
    config_search_concurrency: List[int] = field(default_factory=lambda: [])
    config_search_batch_sizes: List[int] = field(default_factory=lambda: [])
    config_search_instance_counts: Dict[DeviceKind, List] = field(default_factory=lambda: {})
    config_search_max_batch_sizes: List[int] = field(default_factory=lambda: [])
    config_search_preferred_batch_sizes: List[List[int]] = field(default_factory=lambda: [])
    config_search_backend_parameters: Dict[str, List[str]] = field(default_factory=lambda: {})
    config_search_early_exit_enable: bool = False


@dataclass
class ModelAnalyzerAnalysisConfig(BaseConfig):
    top_n_configs: int = 3
    objectives: Dict[str, int] = field(default_factory=lambda: {"perf_throughput": 10})
    max_latency_ms: Optional[int] = None
    min_throughput: int = 0
    max_gpu_usage_mb: Optional[int] = None
