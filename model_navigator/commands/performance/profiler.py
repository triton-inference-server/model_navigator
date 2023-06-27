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

import logging
import pathlib
from typing import List, Optional

import numpy as np
from jsonlines import jsonlines

from model_navigator.api.config import OptimizationProfile, Sample
from model_navigator.commands.performance.results import ProfilingResults
from model_navigator.core.logger import LOGGER
from model_navigator.exceptions import ModelNavigatorError
from model_navigator.runners.base import NavigatorRunner, NavigatorStabilizedRunner
from model_navigator.utils.dataloader import expand_sample


class Profiler:
    """Runs profiling on a runner a profiling sample.

    Example:
        Profiler(
            profile=OptimizationProfile(),
            results_path="results.jsonl",
        ).run(
            runner=runner,
            profiling_sample={"input_1": np.ones(1, 3)}
        )
    """

    def __init__(
        self,
        profile: OptimizationProfile,
        results_path: pathlib.Path,
        batch_dim: Optional[int] = 0,
    ) -> None:
        """Initialize the Profiler.

        Args:
            profile: Optimization profile used for configuration.
            results_path: Jsonlines path to store the results in.
            batch_dim: Batch dimension. Defaults to 0.

        Raises:
            ValueError: When batch_dim is None, but profile.batch_sizes is not None.
        """
        self._profile = profile
        self._batch_dim = batch_dim
        self._results_path = results_path

        if self._batch_dim is None:
            self._batch_sizes = [None]
        else:
            self._batch_sizes = (
                self._profile.batch_sizes
                if self._profile.batch_sizes
                else (2 ** np.arange(31, dtype=np.int32)).tolist()
            )

    def run(
        self,
        runner: NavigatorRunner,
        profiling_sample: Sample,
        sample_id: int,
    ) -> List[ProfilingResults]:
        """Run profiling.

        Args:
            runner: Runner to profile.
            profiling_sample: Sample used for profiling.
            sample_id: Identifier of profiled sample.

        Returns:
            List[ProfilingResults]: Results for each of the batch sizes from profiler configuration.
        """
        results, prev_result = [], None
        with runner:
            for batch_size in self._batch_sizes:
                LOGGER.log(self._profiling_results_logging_level, f"Performance profiling for {runner.name()} started.")
                if batch_size:
                    LOGGER.log(self._profiling_results_logging_level, f"Batch size: {batch_size}.")
                sample = expand_sample(profiling_sample, self._batch_dim, batch_size)
                profiling_result = self._run_measurement(runner, sample, batch_size, sample_id)
                LOGGER.log(
                    self._profiling_results_logging_level,
                    (
                        f"Performance profiling result for {runner.name()} "
                        f"and batch size: {batch_size}:\n{profiling_result}"
                    ),
                )
                with jsonlines.open(self._results_path.as_posix(), "a") as f:
                    f.write(profiling_result.to_dict())

                results.append(profiling_result)
                if prev_result is not None and profiling_result.throughput < prev_result.throughput * (
                    1 + self._profile.throughput_cutoff_threshold
                ):
                    break
                prev_result = profiling_result

        return results

    @property
    def _profiling_results_logging_level(self):
        return logging.INFO

    def _run_window_measurement(
        self, runner: NavigatorRunner, sample: Sample, batch_size: Optional[int], sample_id: int
    ) -> ProfilingResults:
        measurements = []
        for _ in range(self._profile.window_size):
            runner.infer(sample)
            measurements.append(runner.last_inference_time() * 1000)

        return ProfilingResults.from_measurements(measurements, batch_size, sample_id)

    def _is_measurement_stable(self, profiling_results: List[ProfilingResults], count: int = 3) -> bool:
        if len(profiling_results) < count:
            return False

        profiling_results = profiling_results[-count:]
        avg_latencies = [result.avg_latency for result in profiling_results]
        avg_latency = np.mean(avg_latencies)
        deviation_perc = np.abs((avg_latencies - avg_latency) / avg_latency * 100)

        return np.all(deviation_perc < self._profile.stability_percentage)

    def _measurements_result(self, profiling_results: List[ProfilingResults], count: int = 3) -> ProfilingResults:
        if len(profiling_results) < count:
            raise ModelNavigatorError("Measurements results requires at least 3 consecutive stable measurements.")

        profiling_results = profiling_results[-count:]
        profiling_result = ProfilingResults.from_profiling_results(profiling_results)

        return profiling_result

    def _run_measurement(
        self, runner: NavigatorRunner, sample: Sample, batch_size: Optional[int], sample_id: int
    ) -> ProfilingResults:
        profiling_results = []

        if runner.is_stabilized():
            assert isinstance(runner, NavigatorStabilizedRunner)
            runner.infer(sample)
            profiling_result = ProfilingResults.from_stable_runner(runner, batch_size, sample_id)
            return profiling_result
        else:
            for idx in range(self._profile.max_trials):
                profiling_result = self._run_window_measurement(runner, sample, batch_size, sample_id)
                profiling_results.append(profiling_result)
                LOGGER.debug(
                    f"Measurement [{idx}]: {profiling_result.throughput} infer/sec, {profiling_result.avg_latency} ms"
                )
                if self._is_measurement_stable(profiling_results, count=min(3, self._profile.max_trials)):
                    return self._measurements_result(profiling_results, count=min(3, self._profile.max_trials))

        raise RuntimeError(
            "Unable to get stable performance results. Consider increasing "
            "measurement_interval | measurement_request_count | stability_percentage | max_trials in OptimizationProfile."
        )
