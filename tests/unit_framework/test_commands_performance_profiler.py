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
from unittest.mock import MagicMock

from model_navigator.framework_api.commands.performance import Profiler, ProfilerConfig, ProfilingResults


def test_is_measurement_stable_return_false_when_window_is_empty():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        runner=MagicMock(),
        profiling_sample=MagicMock(),
        config=profiler_config,
    )

    assert profiler._is_measurement_stable([]) is False


def test_is_measurement_stable_return_false_when_window_size_less_than_count():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        runner=MagicMock(),
        profiling_sample=MagicMock(),
        config=profiler_config,
    )

    batch_size = 1
    measurements = [25, 24, 23]
    windows = [
        ProfilingResults.from_measurments(measurements, batch_size),
        ProfilingResults.from_measurments(measurements, batch_size),
    ]

    assert profiler._is_measurement_stable(windows) is False


def test_is_measurement_stable_return_false_when_avg_latencies_are_out_of_stability_range():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        runner=MagicMock(),
        profiling_sample=MagicMock(),
        config=profiler_config,
    )

    batch_size = 1
    windows = [
        ProfilingResults.from_measurments([250, 220, 200], batch_size),
        ProfilingResults.from_measurments([200, 150, 100], batch_size),
        ProfilingResults.from_measurments([50, 49, 47], batch_size),
    ]

    assert bool(profiler._is_measurement_stable(windows)) is False


def test_is_measurement_stable_return_true_when_avg_latencies_are_in_stability_range():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        runner=MagicMock(),
        profiling_sample=MagicMock(),
        config=profiler_config,
    )

    batch_size = 1
    windows = [
        ProfilingResults.from_measurments([250, 220, 200], batch_size),
        ProfilingResults.from_measurments([200, 150, 100], batch_size),
        ProfilingResults.from_measurments([52, 52, 51], batch_size),
        ProfilingResults.from_measurments([50, 49, 48], batch_size),
        ProfilingResults.from_measurments([52, 49, 47], batch_size),
    ]

    assert bool(profiler._is_measurement_stable(windows)) is True
