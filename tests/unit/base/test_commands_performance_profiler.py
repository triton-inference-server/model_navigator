# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
import pathlib
import tempfile
from unittest.mock import MagicMock

import numpy as np

from model_navigator.commands.performance.profiler import OptimizationProfile, Profiler, ProfilingResults


def test_batch_size_is_set_correctly_when_no_max_or_batch_sizes_passed():
    optimization_profile = OptimizationProfile()
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._batch_sizes == (2 ** np.arange(31)).tolist()


def test_batch_size_is_set_correctly_when_unsorted_batch_sizes_passed():
    optimization_profile = OptimizationProfile(batch_sizes=[1, 16, 4, 8])
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._batch_sizes == [1, 4, 8, 16]


def test_batch_size_is_set_correctly_when_sorted_batch_sizes_passed():
    optimization_profile = OptimizationProfile(batch_sizes=[1, 4, 16])
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._batch_sizes == [1, 4, 16]


def test_batch_size_is_set_correctly_when_max_batch_size_passed():
    optimization_profile = OptimizationProfile(max_batch_size=1)
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._batch_sizes == [1]

    optimization_profile = OptimizationProfile(max_batch_size=32)
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._batch_sizes == [1, 2, 4, 8, 16, 32]


def test_batch_size_is_set_correctly_when_max_batch_size_which_is_not_power_of_2_passed():
    optimization_profile = OptimizationProfile(max_batch_size=3)
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._batch_sizes == [1, 2, 3]

    optimization_profile = OptimizationProfile(max_batch_size=127)
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._batch_sizes == [1, 2, 4, 8, 16, 32, 64, 127]


def test_is_measurement_stable_return_false_when_window_is_empty():
    optimization_profile = OptimizationProfile(batch_sizes=[1])
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    assert profiler._is_measurement_stable([]) is False


def test_is_measurement_stable_return_false_when_window_size_less_than_count():
    optimization_profile = OptimizationProfile(batch_sizes=[1])
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )
    sample_id = 0
    batch_size = 1
    measurements = [25, 24, 23]
    gpu_clocks = [1500, np.nan]
    windows = [
        ProfilingResults.from_measurements(measurements, gpu_clocks, batch_size, sample_id),
        ProfilingResults.from_measurements(measurements, gpu_clocks, batch_size, sample_id),
    ]

    assert profiler._is_measurement_stable(windows) is False


def test_is_measurement_stable_return_false_when_avg_latencies_are_out_of_stability_range():
    optimization_profile = OptimizationProfile(batch_sizes=[1])
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    sample_id = 0
    batch_size = 1
    windows = [
        ProfilingResults.from_measurements([250, 220, 200], [1500, np.nan], batch_size, sample_id),
        ProfilingResults.from_measurements([200, 150, 100], [1500, np.nan], batch_size, sample_id),
        ProfilingResults.from_measurements([50, 49, 47], [1500, np.nan], batch_size, sample_id),
    ]

    assert bool(profiler._is_measurement_stable(windows)) is False


def test_is_measurement_stable_return_true_when_avg_latencies_are_in_stability_range():
    optimization_profile = OptimizationProfile(batch_sizes=[1])
    profiler = Profiler(
        profile=optimization_profile,
        results_path=MagicMock(),
    )

    sample_id = 0
    batch_size = 1
    windows = [
        ProfilingResults.from_measurements([250, 220, 200], [1500, np.nan], batch_size, sample_id),
        ProfilingResults.from_measurements([200, 150, 100], [1500, np.nan], batch_size, sample_id),
        ProfilingResults.from_measurements([52, 52, 51], [1500, np.nan], batch_size, sample_id),
        ProfilingResults.from_measurements([50, 49, 48], [1500, np.nan], batch_size, sample_id),
        ProfilingResults.from_measurements([52, 49, 47], [1500, np.nan], batch_size, sample_id),
    ]

    assert bool(profiler._is_measurement_stable(windows)) is True


def test_profiler_run_return_batch_sizes_upto_4_when_batch_size_4_saturates_throughput(mocker):
    mocker.patch("model_navigator.utils.dataloader.expand_sample", return_value=MagicMock())
    mocker.patch(
        "model_navigator.commands.performance.Profiler._run_measurement",
        side_effect=[
            ProfilingResults.from_measurements([10, 10, 10], [1500, np.nan], 1, 0),
            ProfilingResults.from_measurements([15, 15, 15], [1500, np.nan], 2, 0),
            ProfilingResults.from_measurements([30, 30, 30], [1500, np.nan], 4, 0),
            ProfilingResults.from_measurements([30, 30, 30], [1500, np.nan], 8, 0),
        ],
    )

    optimization_profile = OptimizationProfile()
    with tempfile.NamedTemporaryFile() as temp:
        profiler = Profiler(
            profile=optimization_profile,
            results_path=pathlib.Path(temp.name),
        )

    results = profiler.run(runner=MagicMock(), profiling_sample=MagicMock(), sample_id=0)

    assert results[-1].batch_size == 4
