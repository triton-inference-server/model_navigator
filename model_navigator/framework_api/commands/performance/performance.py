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

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from polygraphy.backend.base import BaseRunner

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import DataObject, Sample, TensorMetadata
from model_navigator.framework_api.exceptions import ExecutionContext, ModelNavigatorExportAPIError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.runners.base import INavigatorStabilizedRunner
from model_navigator.framework_api.runners.runner_manager import RunnerManager
from model_navigator.framework_api.utils import (
    JitType,
    Parameter,
    RuntimeProvider,
    Status,
    format_to_relative_model_path,
    parse_kwargs_to_cmd,
)
from model_navigator.model import Format

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


@dataclass
class ProfilingResults(DataObject):
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
    def from_dict(cls, d: Mapping):
        return cls(**d)

    @classmethod
    def from_measurements(cls, measurements: List[float], batch_size: int):
        measurements = np.array(measurements)
        return cls(
            batch_size=batch_size,
            avg_latency=float(np.mean(measurements)),
            std_latency=float(np.std(measurements)),
            p50_latency=float(np.percentile(measurements, 50)),
            p90_latency=float(np.percentile(measurements, 90)),
            p95_latency=float(np.percentile(measurements, 95)),
            p99_latency=float(np.percentile(measurements, 99)),
            throughput=float(1000 * max(1, batch_size) / np.mean(measurements)),
            request_count=len(measurements),
        )

    @classmethod
    def from_profiling_results(cls, profiling_results: List["ProfilingResults"]):
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
    def from_stable_runner(cls, runner: INavigatorStabilizedRunner, batch_size: int):

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

    def __str__(self):
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


class MeasurementMode(Parameter):
    TIME_WINDOWS = "time_windows"
    COUNT_WINDOWS = "count_windows"


@dataclass
class ProfilerConfig(DataObject):
    batch_sizes: Optional[Sequence[int]] = None
    measurement_interval: Optional[float] = 5000  # ms
    measurement_mode: MeasurementMode = MeasurementMode.COUNT_WINDOWS
    measurement_request_count: Optional[int] = 50
    stability_percentage: float = 10.0
    max_trials: int = 10

    @classmethod
    def from_dict(cls, dict: Mapping):
        return cls(
            batch_sizes=dict.get("batch_sizes"),
            measurement_interval=dict.get("measurement_interval"),
            measurement_mode=MeasurementMode(dict.get("measurement_mode", MeasurementMode.TIME_WINDOWS)),
            measurement_request_count=dict.get("measurement_request_count"),
            stability_percentage=dict.get("stability_percentage", 10.0),
            max_trials=dict.get("max_trials", 10),
        )


class Profiler:
    def __init__(
        self,
        runner: BaseRunner,
        profiling_sample: Sample,
        config: ProfilerConfig,
        batch_dim: Optional[int] = 0,
        max_batch_size: Optional[int] = None,
    ) -> None:

        self._runner = runner
        self._profiling_sample = profiling_sample
        self._config = config
        self._batch_dim = batch_dim

        if self._batch_dim is None:
            if self._config.batch_sizes:
                raise ValueError(
                    f"Model does not support batching, but profiling batch sizes are {self._config.batch_sizes}."
                )
            self._batch_sizes = [-1]
        else:
            if self._config.batch_sizes:
                self._batch_sizes = self._config.batch_sizes
            else:
                assert max_batch_size
                self._batch_sizes = [1, max_batch_size]

    @staticmethod
    def expand_sample(sample: Sample, axis: Optional[int], n: int):
        if axis is None:
            return sample
        expanded_sample = {}
        for name, tensor in sample.items():
            expanded_sample[name] = tensor.repeat(n, axis=axis)
        return expanded_sample

    def _run_time_window_measurement(self, runner: BaseRunner, sample: Sample, batch_size: int) -> ProfilingResults:
        measurements = []
        start = time.monotonic()
        while (time.monotonic() - start) * 1000 < self._config.measurement_interval:
            runner.infer(sample)
            measurements.append(runner.last_inference_time() * 1000)

        return ProfilingResults.from_measurements(measurements, batch_size)

    def _run_count_window_measurement(self, runner: BaseRunner, sample: Sample, batch_size: int) -> ProfilingResults:
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
            raise ModelNavigatorExportAPIError(
                "Measurements results requires at least 3 consecutive stable measurements."
            )

        profiling_results = profiling_results[-count:]
        profiling_result = ProfilingResults.from_profiling_results(profiling_results)

        return profiling_result

    def _run_measurement(self, runner: BaseRunner, sample: Sample, batch_size: int) -> ProfilingResults:
        profiling_results = []

        measurement_fn = {
            MeasurementMode.TIME_WINDOWS: self._run_time_window_measurement,
            MeasurementMode.COUNT_WINDOWS: self._run_count_window_measurement,
        }[self._config.measurement_mode]

        if runner.is_inference_time_stabilized():
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

    def run(self) -> List[ProfilingResults]:
        results = []
        with self._runner as runner:
            for batch_size in self._batch_sizes:
                sample = self.expand_sample(self._profiling_sample, self._batch_dim, batch_size)
                LOGGER.info(f"Performance profiling for {runner.name} and batch size: {batch_size} started")
                profiling_result = self._run_measurement(runner, sample, batch_size)
                LOGGER.info(
                    f"Performance profiling result for {runner.name} and batch size: {batch_size}:\n{profiling_result}"
                )
                results.append(profiling_result)
        return results


class Performance(Command):
    def __init__(
        self,
        name: str,
        target_format: Format,
        requires: Tuple[Command, ...] = (),
        target_jit_type: Optional[JitType] = None,
        target_precision: Optional[TensorRTPrecision] = None,
        runtime_provider: Optional[RuntimeProvider] = None,
        enable_xla: Optional[bool] = None,
        jit_compile: Optional[bool] = None,
    ):
        super().__init__(
            name=name, command_type=CommandType.PERFORMANCE, target_format=target_format, requires=requires
        )
        self.target_jit_type = target_jit_type
        self.target_precision = target_precision
        self.runtime_provider = runtime_provider
        self.enable_xla = enable_xla
        self.jit_compile = jit_compile

    def _update_package_descriptor(self, package_descriptor: "PackageDescriptor", **kwargs) -> None:
        runtime_results = package_descriptor.get_runtime_results(
            format=self.target_format,
            jit_type=self.target_jit_type,
            precision=self.target_precision,
            runtime_provider=self.runtime_provider,
            enable_xla=self.enable_xla,
            jit_compile=self.jit_compile,
        )
        if runtime_results.status == Status.OK:
            if self.status == Status.OK:
                runtime_results.performance = self.output
            else:
                runtime_results.status = self.status
                runtime_results.err_msg[self.command_type.value] = self.err_msg

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        profiler_config: ProfilerConfig,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: str,
        batch_dim: Optional[int],
        max_batch_size: Optional[int],
        **kwargs,
    ) -> List[ProfilingResults]:
        LOGGER.info(f"Performance test for: {self.target_format} {self.runtime_provider} started.")

        model_path = workdir / format_to_relative_model_path(
            format=self.target_format,
            jit_type=self.target_jit_type,
            precision=self.target_precision,
            enable_xla=self.enable_xla,
            jit_compile=self.jit_compile,
        )
        model_dir = model_path.parent

        runner_manager = RunnerManager(input_metadata, output_metadata, target_device)

        with ExecutionContext(
            workdir=workdir,
            script_path=model_dir / "reproduce_profiling.py",
            cmd_path=model_dir / "reproduce_profiling.sh",
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "navigator_workdir": workdir.as_posix(),
                "model_name": model_name,
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "format": self.target_format.value,
                "precision": self.target_precision.value if self.target_precision else None,
                "jit_type": self.target_jit_type.value if self.target_jit_type else None,
                "runtime": self.runtime_provider.value if self.runtime_provider else None,
                "enable_xla": self.enable_xla,
                "jit_compile": self.jit_compile,
                "profiler_config": profiler_config.to_dict(parse=True),
                "max_batch_size": max_batch_size,
                "runner_manager_dict": runner_manager.to_dict(parse=True),
            }

            args = parse_kwargs_to_cmd(kwargs, (list, dict, tuple))

            from model_navigator.framework_api.commands.performance import performance_script

            context.execute_external_runtime_script(performance_script.__file__, args)
            results = [ProfilingResults.from_dict(res) for res in json.load(temp_file)]

        return results
