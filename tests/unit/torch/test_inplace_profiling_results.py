# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np

from model_navigator.inplace.profiling import ProfilingResult


def test_inplace_profiling_result_calculates_correct_statistics_from_measurements():
    measurements = [2, 4, 6, 8]
    batch_size = 1
    profiling_result = ProfilingResult.from_measurements(measurements, batch_size=1)

    atol = 0.0001
    assert np.isclose(profiling_result.avg_latency, np.mean(measurements), atol=atol)
    assert np.isclose(profiling_result.std_latency, np.std(measurements), atol=atol)
    assert np.isclose(profiling_result.p50_latency, np.percentile(measurements, 50), atol=atol)
    assert np.isclose(profiling_result.p90_latency, np.percentile(measurements, 90), atol=atol)
    assert np.isclose(profiling_result.p95_latency, np.percentile(measurements, 95), atol=atol)
    assert np.isclose(profiling_result.p99_latency, np.percentile(measurements, 99), atol=atol)
    assert np.isclose(profiling_result.throughput, 1000 * batch_size / np.mean(measurements), atol=atol)
    assert np.isclose(profiling_result.throughput, 200.0, atol=atol)
    assert profiling_result.request_count
    assert profiling_result.batch_size == 1
    assert profiling_result.avg_gpu_clock is None


def test_inplace_profiling_result_calculates_correct_statistics_from_profiling_result():
    trials = [[1, 2, 3, 4], [11, 22, 33, 44]]
    batch_size = 1
    profiling_results = [ProfilingResult.from_measurements(trial, batch_size) for trial in trials]

    profiling_result = ProfilingResult.from_profiling_results(profiling_results)
    atol = 0.01
    assert np.isclose(profiling_result.avg_latency, np.mean([r.avg_latency for r in profiling_results]), atol=atol)
    assert np.isclose(profiling_result.std_latency, np.std([r.std_latency for r in profiling_results]), atol=atol)
    assert np.isclose(
        profiling_result.p50_latency, np.percentile([r.p50_latency for r in profiling_results], 50), atol=atol
    )
    assert np.isclose(
        profiling_result.p90_latency, np.percentile([r.p90_latency for r in profiling_results], 90), atol=atol
    )
    assert np.isclose(
        profiling_result.p95_latency, np.percentile([r.p95_latency for r in profiling_results], 95), atol=atol
    )
    assert np.isclose(
        profiling_result.p99_latency, np.percentile([r.p99_latency for r in profiling_results], 99), atol=atol
    )
    assert np.isclose(
        profiling_result.throughput, 1000 * batch_size / np.mean([r.avg_latency for r in profiling_results]), atol=atol
    )
    assert profiling_result.request_count
    assert profiling_result.batch_size == 1
    assert profiling_result.avg_gpu_clock is None
