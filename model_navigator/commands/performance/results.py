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

import collections
import dataclasses
import warnings
from typing import Dict, List, Mapping, Optional

import numpy as np

from model_navigator.runners.base import InferenceStep, InferenceTime, NavigatorStabilizedRunner
from model_navigator.utils.common import DataObject


@dataclasses.dataclass
class ProfilingStepResults(DataObject):
    """Profiling step results."""

    avg_time: float  # ms
    std_time: float  # ms
    p50_time: float  # ms
    p90_time: float  # ms
    p95_time: float  # ms
    p99_time: float  # ms

    @classmethod
    def from_dict(cls, d: Mapping) -> "ProfilingStepResults":
        """Instantiate ProfilingStepResults from a json dictionary.

        Args:
            d (Mapping): Data dictionary.

        Returns:
            ProfilingStepResults
        """
        return cls(
            avg_time=d["avg_time"],
            std_time=d["std_time"],
            p50_time=d["p50_time"],
            p90_time=d["p90_time"],
            p95_time=d["p95_time"],
            p99_time=d["p99_time"],
        )


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

    detailed_results: Dict[str, ProfilingStepResults] = dataclasses.field(default_factory=dict)

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

        detailed_results = {}
        if "detailed_results" in d:
            assert isinstance(d["detailed_results"], dict)
            for step_name, step_dict in d["detailed_results"].items():
                detailed_results[step_name] = ProfilingStepResults.from_dict(step_dict)

        return cls(
            sample_id=d["sample_id"],
            batch_size=d.get("batch_size"),
            request_count=d["request_count"],
            avg_gpu_clock=d.get("avg_gpu_clock"),
            avg_latency=d["avg_latency"],
            std_latency=d["std_latency"],
            p50_latency=d["p50_latency"],
            p90_latency=d["p90_latency"],
            p95_latency=d["p95_latency"],
            p99_latency=d["p99_latency"],
            throughput=d["throughput"],
            detailed_results=detailed_results,
        )

    @classmethod
    def from_measurements(
        cls, measurements: List[InferenceTime], gpu_clocks: List[float], batch_size: Optional[int], sample_id: int
    ) -> "ProfilingResults":
        """Instantiate ProfilingResults from a list of measurements.

        Args:
            measurements: List of measurements.
            gpu_clocks: List of GPU clocks.
            batch_size: Batch size.
            sample_id: Sample id

        Returns:
            ProfilingResults
        """
        measurements = np.array(measurements)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_gpu_clock = np.nanmean([gpu_clock for gpu_clock in gpu_clocks if gpu_clock is not None])

        step_measurements: Dict[str, List[float]] = collections.defaultdict(list)
        for measurement in measurements:
            for step_name, step_measurement in measurement.items():
                step_measurements[step_name].append(step_measurement)

        detailed_results = {
            step_name: ProfilingStepResults(
                avg_time=float(np.mean(detailed_results)),
                std_time=float(np.std(detailed_results)),
                p50_time=float(np.percentile(detailed_results, 50)),
                p90_time=float(np.percentile(detailed_results, 90)),
                p95_time=float(np.percentile(detailed_results, 95)),
                p99_time=float(np.percentile(detailed_results, 99)),
            )
            for step_name, detailed_results in step_measurements.items()
        }

        assert InferenceStep.TOTAL.value in detailed_results
        return cls(
            sample_id=sample_id,
            batch_size=batch_size,
            avg_gpu_clock=float(avg_gpu_clock),
            request_count=len(measurements),
            detailed_results=detailed_results,
            avg_latency=detailed_results[InferenceStep.TOTAL.value].avg_time,
            std_latency=detailed_results[InferenceStep.TOTAL.value].std_time,
            p50_latency=detailed_results[InferenceStep.TOTAL.value].p50_time,
            p90_latency=detailed_results[InferenceStep.TOTAL.value].p90_time,
            p95_latency=detailed_results[InferenceStep.TOTAL.value].p95_time,
            p99_latency=detailed_results[InferenceStep.TOTAL.value].p99_time,
            throughput=(1000 * (batch_size or 1) / detailed_results[InferenceStep.TOTAL.value].avg_time),
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

        step_measurements: Dict[str, List[ProfilingStepResults]] = collections.defaultdict(list)
        for result in profiling_results:
            assert result.batch_size == batch_size, "Batch size must be the same for all profiling results"
            for step_name, step_result in result.detailed_results.items():
                step_measurements[step_name].append(step_result)

        detailed_results = {
            step_name: ProfilingStepResults(
                avg_time=float(np.mean([result.avg_time for result in detailed_results])),
                std_time=float(np.std([result.std_time for result in detailed_results])),
                p50_time=float(np.percentile([result.p50_time for result in detailed_results], 50)),
                p90_time=float(np.percentile([result.p90_time for result in detailed_results], 90)),
                p95_time=float(np.percentile([result.p95_time for result in detailed_results], 95)),
                p99_time=float(np.percentile([result.p99_time for result in detailed_results], 99)),
            )
            for step_name, detailed_results in step_measurements.items()
        }

        assert InferenceStep.TOTAL.value in detailed_results
        return cls(
            sample_id=profiling_results[0].sample_id,
            batch_size=batch_size,
            avg_gpu_clock=float(avg_gpu_clock),
            request_count=int(np.mean([result.request_count for result in profiling_results])),
            detailed_results=detailed_results,
            avg_latency=detailed_results[InferenceStep.TOTAL.value].avg_time,
            std_latency=detailed_results[InferenceStep.TOTAL.value].std_time,
            p50_latency=detailed_results[InferenceStep.TOTAL.value].p50_time,
            p90_latency=detailed_results[InferenceStep.TOTAL.value].p90_time,
            p95_latency=detailed_results[InferenceStep.TOTAL.value].p95_time,
            p99_latency=detailed_results[InferenceStep.TOTAL.value].p99_time,
            throughput=(1000 * (batch_size or 1) / detailed_results[InferenceStep.TOTAL.value].avg_time),
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
