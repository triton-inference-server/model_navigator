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
"""Runners profiling."""
import dataclasses
import warnings
from typing import List, Mapping, Optional

import numpy as np

from model_navigator.runners.base import NavigatorStabilizedRunner
from model_navigator.utils.common import DataObject


@dataclasses.dataclass
class ProfilingResults(DataObject):
    """Profiling results."""

    sample_id: int
    batch_size: Optional[int]
    avg_latency: float  # ms
    std_latency: float  # ms
    p50_latency: float  # ms
    p90_latency: float  # ms
    p95_latency: float  # ms
    p99_latency: float  # ms
    throughput: float  # infer / sec
    request_count: int
    avg_gpu_clock: Optional[float] = None  # MHz

    @classmethod
    def from_dict(cls, d: Mapping) -> "ProfilingResults":
        """Instantiate ProfilingResults from a json dictionary.

        Args:
            d (Mapping): Data dictionary.

        Returns:
            ProfilingResults
        """
        if "sample_id" not in d:
            d = {"sample_id": 0, **d}

        return cls(**d)

    @classmethod
    def from_measurements(
        cls, measurements: List[float], gpu_clocks: List[float], batch_size: Optional[int], sample_id: int
    ) -> "ProfilingResults":
        """Instantiate ProfilingResults from a list of measurements.

        Args:
            measurements: List of measurements.
            batch_size: Batch size.
            sample_id: Sample id

        Returns:
            ProfilingResults
        """
        measurements = np.array(measurements)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_gpu_clock = np.nanmean([gpu_clock for gpu_clock in gpu_clocks if gpu_clock is not None])
        return cls(
            sample_id=sample_id,
            batch_size=batch_size,
            avg_latency=float(np.mean(measurements)),
            std_latency=float(np.std(measurements)),
            p50_latency=float(np.percentile(measurements, 50)),
            p90_latency=float(np.percentile(measurements, 90)),
            p95_latency=float(np.percentile(measurements, 95)),
            p99_latency=float(np.percentile(measurements, 99)),
            throughput=float(1000 * (batch_size or 1) / np.mean(measurements)),
            avg_gpu_clock=float(avg_gpu_clock),
            request_count=len(measurements),
        )

    @classmethod
    def from_profiling_results(cls, profiling_results: List["ProfilingResults"]) -> "ProfilingResults":
        """Instantiate ProfilingResults as a mean of other profiling results.

        Args:
            profiling_results (List[ProfilingResults]): List of profiling results to average.

        Returns:
            ProfilingResults
        """
        batch_size = profiling_results[0].batch_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_gpu_clock = np.nanmean([result.avg_gpu_clock for result in profiling_results])

        return cls(
            sample_id=profiling_results[0].sample_id,
            batch_size=batch_size,
            avg_latency=float(np.mean([result.avg_latency for result in profiling_results])),
            std_latency=float(np.mean([result.std_latency for result in profiling_results])),
            p50_latency=float(np.mean([result.p50_latency for result in profiling_results])),
            p90_latency=float(np.mean([result.p90_latency for result in profiling_results])),
            p95_latency=float(np.mean([result.p95_latency for result in profiling_results])),
            p99_latency=float(np.mean([result.p99_latency for result in profiling_results])),
            throughput=float(np.mean([result.throughput for result in profiling_results])),
            avg_gpu_clock=float(avg_gpu_clock),
            request_count=int(np.mean([result.request_count for result in profiling_results])),
        )

    @classmethod
    def from_stable_runner(
        cls, runner: NavigatorStabilizedRunner, batch_size: int, sample_id: int
    ) -> "ProfilingResults":
        """Instantiate ProfilingResults from a stable runner results.

        Args:
            runner: Stable runner instance.
            batch_size: Batch size.
            sample_id: Identifier of sample

        Returns:
            ProfilingResults
        """
        return cls(
            sample_id=sample_id,
            batch_size=batch_size,
            avg_latency=runner.avg_latency(),
            std_latency=runner.std_latency(),
            p50_latency=runner.p50_latency(),
            p90_latency=runner.p90_latency(),
            p95_latency=runner.p95_latency(),
            p99_latency=runner.p99_latency(),
            throughput=float(1000 * max(1, batch_size) / runner.avg_latency()),
            avg_gpu_clock=runner.avg_gpu_clock(),
            request_count=runner.request_count(),
        )

    def __str__(self) -> str:
        """Get string representation."""
        avg_gpu_clock = f"{self.avg_gpu_clock:.4f}" if self.avg_gpu_clock is not None else "-"
        return (
            f"Sample ID: {self.sample_id}\n"
            f"Batch: {self.batch_size}\n"
            f"Request count: {self.request_count}\n"
            f"Throughput: {self.throughput:.4f} [infer/sec]\n"
            f"Avg Latency: {self.avg_latency:.4f} [ms]\n"
            f"Std Latency: {self.std_latency:.4f} [ms]\n"
            f"p50 Latency: {self.p50_latency:.4f} [ms]\n"
            f"p90 Latency: {self.p90_latency:.4f} [ms]\n"
            f"p95 Latency: {self.p95_latency:.4f} [ms]\n"
            f"p99 Latency: {self.p99_latency:.4f} [ms]\n"
            f"Avg GPU clock: {avg_gpu_clock} [MHz]"
        )
