# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Profiling results generated from profile method."""

import dataclasses
import pathlib
from typing import Dict, List, Union

import yaml

from model_navigator.commands.base import CommandStatus
from model_navigator.utils.common import dataclass2dict


@dataclasses.dataclass
class ProfilingResult:
    """Result for single profiling for sample.

    Args:
        batch_size: Size of batch used for profiling
        avg_latency: Average latency of profiling
        std_latency: Standard deviation of profiled latency
        p50_latency: 50th percentile of measured latency
        p90_latency: 90th percentile of measured latency
        p95_latency: 95th percentile of measured latency
        p99_latency: 99th percentile of measured latency
        throughput: Inferences per second
        request_count: Number of inference requests
    """

    batch_size: int
    avg_latency: float  # ms
    std_latency: float  # ms
    p50_latency: float  # ms
    p90_latency: float  # ms
    p95_latency: float  # ms
    p99_latency: float  # ms
    throughput: float  # infer / sec
    avg_gpu_clock: float  # MHz
    request_count: int


@dataclasses.dataclass
class RunnerProfilingResults:
    """Profiling results for runner.

    Args:
        status: Status of profiling execution
        detailed: Result mapping - per sample id
    """

    status: CommandStatus
    detailed: Dict[int, List[ProfilingResult]]


@dataclasses.dataclass
class RunnerResults:
    """Result for runners.

    Args:
        runners: Mapping of runner and their profiling results
    """

    runners: Dict[str, RunnerProfilingResults]


@dataclasses.dataclass
class ProfilingResults:
    """Profiling results for models.

    Args:
        models: Mapping of models and their runner results
    """

    models: Dict[str, RunnerResults]
    samples_data: Dict[int, Dict]

    def to_dict(self):
        """Return results in form of dictionary."""
        return dataclass2dict(self)

    def to_file(self, path: Union[str, pathlib.Path]):
        """Save results to file.

        Args:
            path: A path to yaml files
        """
        path = pathlib.Path(path)
        data = self.to_dict()
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
