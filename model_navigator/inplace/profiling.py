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
"""Profiling functionalities."""

import dataclasses
import pathlib
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import yaml

from model_navigator.commands.base import CommandStatus
from model_navigator.commands.performance.nvml_handler import NvmlHandler
from model_navigator.commands.performance.utils import is_measurement_stable
from model_navigator.core.logger import LOGGER
from model_navigator.exceptions import ModelNavigatorError
from model_navigator.utils.common import DataObject


@dataclasses.dataclass
class ProfilingResult(DataObject):
    """Profiling results."""

    batch_size: int
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
    def from_measurements(
        cls, measurements: List[float], batch_size: int, gpu_clocks: Optional[List[float]] = None
    ) -> "ProfilingResult":
        """Create profiling results from measurements.

        Args:
            measurements: List of time measurements.
            batch_size: Batch size.
            gpu_clocks: List of GPU clocks.

        Returns:
            Profiling result.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if gpu_clocks:
                avg_gpu_clock = float(np.nanmean([gpu_clock for gpu_clock in gpu_clocks if gpu_clock is not None]))
            else:
                avg_gpu_clock = None

        avg_latency = float(np.mean(measurements))
        return cls(
            avg_latency=avg_latency,
            std_latency=float(np.std(measurements)),
            p50_latency=float(np.percentile(measurements, 50)),
            p90_latency=float(np.percentile(measurements, 90)),
            p95_latency=float(np.percentile(measurements, 95)),
            p99_latency=float(np.percentile(measurements, 99)),
            throughput=(1000 * (batch_size or 1) / avg_latency),
            batch_size=batch_size,
            request_count=len(measurements),
            avg_gpu_clock=avg_gpu_clock,
        )

    @classmethod
    def from_profiling_results(cls, profiling_results: List["ProfilingResult"]) -> "ProfilingResult":
        """Create profiling results from list of profiling results.

        Args:
            profiling_results: List of profiling results.

        Returns:
            Profiling result.
        """
        batch_size = profiling_results[0].batch_size
        assert all(
            result.batch_size == batch_size for result in profiling_results
        ), "Batch size must be the same for all profiling results"
        if profiling_results[0].avg_gpu_clock:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_gpu_clock = float(np.nanmean([result.avg_gpu_clock for result in profiling_results]))
        else:
            avg_gpu_clock = None

        avg_latency = float(np.mean([result.avg_latency for result in profiling_results]))
        return cls(
            avg_latency=avg_latency,
            std_latency=float(np.std([result.std_latency for result in profiling_results])),
            p50_latency=float(np.percentile([result.p50_latency for result in profiling_results], 50)),
            p90_latency=float(np.percentile([result.p90_latency for result in profiling_results], 90)),
            p95_latency=float(np.percentile([result.p95_latency for result in profiling_results], 95)),
            p99_latency=float(np.percentile([result.p99_latency for result in profiling_results], 99)),
            throughput=(1000 * (batch_size or 1) / avg_latency),
            batch_size=batch_size,
            request_count=int(np.mean([result.request_count for result in profiling_results])),
            avg_gpu_clock=avg_gpu_clock,
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "ProfilingResult":
        """Create profiling results from dict.

        Args:
            data: Profiling results data.

        Returns:
            ProfilingResult: Profiling results.

        """
        return cls(**data)

    def __str__(self) -> str:
        """Get string representation."""
        avg_gpu_clock = f"{self.avg_gpu_clock:.4f}" if self.avg_gpu_clock is not None else "-"
        return (
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


@dataclasses.dataclass
class RunnerProfilingResults(DataObject):
    """Profiling results for runner.

    Args:
        status: Status of profiling execution
        detailed: Result mapping - per sample id
    """

    status: CommandStatus = CommandStatus.INITIALIZED
    detailed: Dict[int, ProfilingResult] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RunnerResults(DataObject):
    """Result for runners.

    Args:
        runners: Mapping of runner and their profiling results
    """

    runners: Dict[str, RunnerProfilingResults] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ProfilingResults(DataObject):
    """Profiling results for models.

    Args:
        models: Mapping of models and their runner results
    """

    models: Dict[str, RunnerResults] = dataclasses.field(default_factory=dict)

    # samples_data: Dict[int, Dict] = dataclasses.field(default_factory=dict) TODO: enable when sample collecting is implemented

    def to_file(self, path: Union[str, pathlib.Path]):
        """Save results to file.

        Args:
            path: A path to yaml files
        """
        path = pathlib.Path(path)
        data = self.to_dict(parse=True)
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


def _run_window_measurement(
    func: Callable,
    sample: Any,
    batch_size: int,
    nvml_handler: NvmlHandler,
    window_size: int,
) -> ProfilingResult:
    if not isinstance(sample, (list, tuple)):
        sample = (sample,)
    if not isinstance(sample[-1], dict):
        sample = (*sample, {})
    *args, kwargs = sample

    measurements = []
    gpu_clocks = []
    for _ in range(window_size):
        start = time.monotonic()
        func(*args, **kwargs)
        end = time.monotonic()
        gpu_clocks.append(nvml_handler.gpu_clock)
        measurements.append((end - start) * 1000.0)  # ms

    return ProfilingResult.from_measurements(measurements=measurements, batch_size=batch_size, gpu_clocks=gpu_clocks)


def _measurements_result(profiling_results: List[ProfilingResult], last_n: int = 3) -> ProfilingResult:
    if len(profiling_results) < last_n:
        raise ModelNavigatorError(f"Measurements results requires at least {last_n} consecutive stable measurements.")

    profiling_results = profiling_results[-last_n:]
    return ProfilingResult.from_profiling_results(profiling_results)


def run_measurement(
    func: Callable,
    sample: Any,
    nvml_handler: NvmlHandler,
    min_trials: int,
    max_trials: int,
    stabilization_windows: int,
    window_size: int,
    stability_percentage: float,
) -> ProfilingResult:
    """Run profiling measurement.

    Args:
        func: Function to profile.
        sample: Sample for the function.
        nvml_handler: NVML handler.
        min_trials: Minimum number of trials.
        max_trials: Maximum number of trials.
        stabilization_windows: Number of stabilization windows.
        window_size: Number of inference queries performed in measurement window
        stability_percentage: Allowed percentage of variation from the mean in three consecutive windows.

    Returns:
        ProfilingResult: Profiling results.
    """
    profiling_results = []
    batch_size = sample[0]
    _, sample = sample  # skip first element - batch_size
    for idx in range(max_trials):
        measurement_id = idx + 1
        profiling_result = _run_window_measurement(
            func=func, sample=sample, batch_size=batch_size, window_size=window_size, nvml_handler=nvml_handler
        )

        profiling_results.append(profiling_result)
        LOGGER.debug(f"Measurement [{measurement_id}], avg_latency: {profiling_result.avg_latency} ms")

        last_n = min(stabilization_windows, max_trials)
        is_stable = is_measurement_stable(profiling_results, last_n=last_n, stability_percentage=stability_percentage)
        if measurement_id >= min_trials and is_stable:
            return _measurements_result(profiling_results, last_n)

    raise RuntimeError(
        "Unable to get stable performance results. Consider increasing "
        "window_size | stability_percentage | max_trials"
    )
