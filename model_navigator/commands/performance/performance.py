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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Type

import numpy as np
from jsonlines import jsonlines

from model_navigator.api.config import Format, MeasurementMode, ProfilerConfig, Sample
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.core.tensor import TensorMetadata
from model_navigator.exceptions import ModelNavigatorError, ModelNavigatorProfilingError
from model_navigator.execution_context import ExecutionContext
from model_navigator.logger import LOGGER
from model_navigator.runners.base import NavigatorRunner, NavigatorStabilizedRunner
from model_navigator.utils.common import DataObject, parse_kwargs_to_cmd
from model_navigator.utils.dataloader import expand_sample
from model_navigator.utils.format_helpers import is_source_format


@dataclass
class ProfilingResults(DataObject):
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

    @classmethod
    def from_dict(cls, d: Mapping) -> "ProfilingResults":
        """Instantiate ProfilingResults from a json dictionary.

        Args:
            d (Mapping): Data dictionary.

        Returns:
            ProfilingResults
        """
        return cls(**d)

    @classmethod
    def from_measurements(cls, measurements: List[float], batch_size: Optional[int]) -> "ProfilingResults":
        """Instantiate ProfilingResults from a list of measurements.

        Args:
            measurements (List[float]): List of measurements.
            batch_size (int): Batch size.

        Returns:
            ProfilingResults
        """
        measurements = np.array(measurements)
        return cls(
            batch_size=batch_size,
            avg_latency=float(np.mean(measurements)),
            std_latency=float(np.std(measurements)),
            p50_latency=float(np.percentile(measurements, 50)),
            p90_latency=float(np.percentile(measurements, 90)),
            p95_latency=float(np.percentile(measurements, 95)),
            p99_latency=float(np.percentile(measurements, 99)),
            throughput=float(1000 * (batch_size or 1) / np.mean(measurements)),
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

        return cls(
            batch_size=batch_size,
            avg_latency=float(np.mean([result.avg_latency for result in profiling_results])),
            std_latency=float(np.mean([result.std_latency for result in profiling_results])),
            p50_latency=float(np.mean([result.p50_latency for result in profiling_results])),
            p90_latency=float(np.mean([result.p90_latency for result in profiling_results])),
            p95_latency=float(np.mean([result.p95_latency for result in profiling_results])),
            p99_latency=float(np.mean([result.p99_latency for result in profiling_results])),
            throughput=float(np.mean([result.throughput for result in profiling_results])),
            request_count=int(np.mean([result.request_count for result in profiling_results])),
        )

    @classmethod
    def from_stable_runner(cls, runner: NavigatorStabilizedRunner, batch_size: int) -> "ProfilingResults":
        """Instantiate ProfilingResults from a stable runner results.

        Args:
            runner (NavigatorStabilizedRunner): Stable runner instance.
            batch_size (int): Batch size.

        Returns:
            ProfilingResults
        """
        return cls(
            batch_size=batch_size,
            avg_latency=runner.avg_latency(),
            std_latency=runner.std_latency(),
            p50_latency=runner.p50_latency(),
            p90_latency=runner.p90_latency(),
            p95_latency=runner.p95_latency(),
            p99_latency=runner.p99_latency(),
            throughput=float(1000 * max(1, batch_size) / runner.avg_latency()),
            request_count=runner.request_count(),
        )

    def __str__(self) -> str:
        """Get string representation."""
        return (
            f"Batch: {self.batch_size}\n"
            f"Request count: {self.request_count}\n"
            f"Throughput: {self.throughput:.4f} [infer/sec]\n"
            f"Avg Latency: {self.avg_latency:.4f} [ms]\n"
            f"Std Latency: {self.std_latency:.4f} [ms]\n"
            f"p50 Latency: {self.p50_latency:.4f} [ms]\n"
            f"p90 Latency: {self.p90_latency:.4f} [ms]\n"
            f"p95 Latency: {self.p95_latency:.4f} [ms]\n"
            f"p99 Latency: {self.p99_latency:.4f} [ms]"
        )


class Profiler:
    """Runs profiling on a runner an profiling sample.

    Example:
        Profiler(
            config=ProfilerConfig(),
            results_path="results.jsonl",
        ).run(
            runner=runner,
            profiling_sample={"input_1": np.ones(1, 3)}
        )
    """

    def __init__(
        self,
        config: ProfilerConfig,
        results_path: Path,
        batch_dim: Optional[int] = 0,
    ) -> None:
        """Initialize the Profiler.

        Args:
            config (ProfilerConfig): Profiler configuration.
            results_path (Path): Jsonlines path to store the results in.
            batch_dim (Optional[int], optional): Batch dimension. Defaults to 0.

        Raises:
            ValueError: When batch_dim is None, but config.batch_sizes is not None.
        """
        self._config = config
        self._batch_dim = batch_dim
        self._results_path = results_path

        if self._batch_dim is None:
            self._batch_sizes = [None]
        else:
            self._batch_sizes = (
                self._config.batch_sizes if self._config.batch_sizes else (2 ** np.arange(31, dtype=np.int32)).tolist()
            )

    @property
    def _profiling_results_logging_level(self):
        return logging.INFO

    def _run_time_window_measurement(
        self, runner: NavigatorRunner, sample: Sample, batch_size: int
    ) -> ProfilingResults:
        measurements = []
        start = time.monotonic()
        while (time.monotonic() - start) * 1000 < self._config.measurement_interval:
            runner.infer(sample)
            measurements.append(runner.last_inference_time() * 1000)

        return ProfilingResults.from_measurements(measurements, batch_size)

    def _run_count_window_measurement(
        self, runner: NavigatorRunner, sample: Sample, batch_size: Optional[int]
    ) -> ProfilingResults:
        measurements = []
        for _ in range(self._config.measurement_request_count):
            runner.infer(sample)
            measurements.append(runner.last_inference_time() * 1000)

        return ProfilingResults.from_measurements(measurements, batch_size)

    def _is_measurement_stable(self, profiling_results: List[ProfilingResults], count: int = 3) -> bool:
        if len(profiling_results) < count:
            return False

        profiling_results = profiling_results[-count:]
        avg_latencies = [result.avg_latency for result in profiling_results]
        avg_latency = np.mean(avg_latencies)
        deviation_perc = np.abs((avg_latencies - avg_latency) / avg_latency * 100)

        return np.all(deviation_perc < self._config.stability_percentage)

    def _measurements_result(self, profiling_results: List[ProfilingResults], count: int = 3) -> ProfilingResults:
        if len(profiling_results) < count:
            raise ModelNavigatorError("Measurements results requires at least 3 consecutive stable measurements.")

        profiling_results = profiling_results[-count:]
        profiling_result = ProfilingResults.from_profiling_results(profiling_results)

        return profiling_result

    def _run_measurement(self, runner: NavigatorRunner, sample: Sample, batch_size: Optional[int]) -> ProfilingResults:
        profiling_results = []

        measurement_fn = {
            MeasurementMode.TIME_WINDOWS: self._run_time_window_measurement,
            MeasurementMode.COUNT_WINDOWS: self._run_count_window_measurement,
        }[self._config.measurement_mode]

        if runner.is_stabilized():
            assert isinstance(runner, NavigatorStabilizedRunner)
            runner.infer(sample)
            profiling_result = ProfilingResults.from_stable_runner(runner, batch_size)
            return profiling_result
        else:
            for idx in range(self._config.max_trials):
                profiling_result = measurement_fn(runner, sample, batch_size)
                profiling_results.append(profiling_result)
                LOGGER.debug(
                    f"Measurement [{idx}]: {profiling_result.throughput} infer/sec, {profiling_result.avg_latency} ms"
                )
                if self._is_measurement_stable(profiling_results, count=min(3, self._config.max_trials)):
                    return self._measurements_result(profiling_results, count=min(3, self._config.max_trials))

        raise RuntimeError(
            "Unable to get stable performance results. Consider increasing "
            "measurement_interval | measurement_request_count | stability_percentage | max_trials in ProfilerConfig."
        )

    def run(
        self,
        runner: NavigatorRunner,
        profiling_sample: Sample,
    ) -> List[ProfilingResults]:
        """Run profiling.

        Args:
            runner (NavigatorRunner): Runner to profile.
            profiling_sample (Sample): Sample used for profiling.

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
                profiling_result = self._run_measurement(runner, sample, batch_size)
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
                    1 + self._config.throughput_cutoff_threshold
                ):
                    break
                prev_result = profiling_result

        return results


class Performance(Command):
    """Performance command."""

    def _run(
        self,
        workspace: Path,
        path: Path,
        format: Format,
        profiler_config: ProfilerConfig,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        verbose: bool,
        runner_cls: Type[NavigatorRunner],
        reproduce_script_dir: Optional[Path] = None,
        model: Optional[Any] = None,
    ) -> CommandOutput:
        """Run performance command.

        Args:
            workspace (Path): Model Navigator workspace path.
            path (Path): Model path, relative to the workspace.
            format (Format): Model format.
            profiler_config (ProfilerConfig): Profiler configuration.
            input_metadata (TensorMetadata): Input metadata.
            output_metadata (TensorMetadata): Output metadata.
            batch_dim (Optional[int]): Batch dimension.
            verbose (bool): If True verbose logging.
            runner_cls (Type[NavigatorRunner]): Runner type to profile the model with.
            reproduce_script_dir (Optional[Path], optional): Path to store the reproducting scripts for the command.
                When None use model directory. Defaults to None.
            model (Optional[Any], optional): Model when profiling on a source format. Defaults to None.

        Returns:
            CommandOutput: Output of the command containing profiling results.
        """
        model_path = workspace / path
        model_dir = model_path.parent
        reproduce_script_dir = reproduce_script_dir or model_dir

        if not is_source_format(format) and not model_path.exists():
            LOGGER.warning(f"Model: {model_path.as_posix()!r} not found, command skipped.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        with ExecutionContext(
            workspace=workspace,
            script_path=reproduce_script_dir / "reproduce_profiling.py",
            cmd_path=reproduce_script_dir / "reproduce_profiling.sh",
            verbose=verbose,
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "navigator_workspace": workspace.as_posix(),
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "runner_name": runner_cls.name(),
                "profiler_config": profiler_config.to_dict(parse=True),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
            }

            args = parse_kwargs_to_cmd(kwargs)

            from model_navigator.commands.performance import performance_script

            if is_source_format(format):
                performance_script.get_model = lambda: model
                args = parse_kwargs_to_cmd(kwargs)
                context.execute_local_runtime_script(
                    performance_script.__file__, performance_script.profile, args, allow_failure=True
                )
            else:
                kwargs["model_path"] = path
                args = parse_kwargs_to_cmd(kwargs)
                context.execute_external_runtime_script(performance_script.__file__, args, allow_failure=True)
            with jsonlines.open(temp_file.name, "r") as f:
                profiling_results = [ProfilingResults.from_dict(res) for res in f]
        if not profiling_results:
            raise ModelNavigatorProfilingError("No profiling results found.")
        return CommandOutput(status=CommandStatus.OK, output={"profiling_results": profiling_results})
