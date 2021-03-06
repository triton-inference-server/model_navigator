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
from enum import Enum

from model_navigator.utils.config import BaseConfig


class MeasurementMode(Enum):
    """
    Measurement modes for
    """

    COUNT_WINDOWS = "count_windows"
    TIME_WINDOWS = "time_windows"


class SharedMemoryMode(Enum):
    """
    Available offline mode for memory
    """

    NONE = "none"
    SYSTEM = "system"
    CUDA = "cuda"


@dataclass
class PerfMeasurementConfig(BaseConfig):
    perf_analyzer_timeout: int = 600
    perf_analyzer_path: str = "perf_analyzer"
    perf_measurement_mode: str = MeasurementMode.COUNT_WINDOWS.value
    perf_measurement_request_count: int = 50
    perf_measurement_interval: int = 5000
    perf_measurement_shared_memory: str = SharedMemoryMode.NONE.value
    perf_measurement_output_shared_memory_size: int = 102400
