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
"""RuntimeAnalyzer class module."""

import dataclasses
from math import inf
from typing import Dict, Optional, Sequence

from model_navigator.commands.correctness.correctness import Correctness
from model_navigator.commands.performance.performance import Performance
from model_navigator.core.logger import LOGGER
from model_navigator.exceptions import ModelNavigatorRuntimeAnalyzerError, ModelNavigatorUserInputError
from model_navigator.package.status import CommandStatus, ModelStatus, RunnerStatus
from model_navigator.runtime_analyzer.strategy import (
    MaxThroughputAndMinLatencyStrategy,
    MaxThroughputStrategy,
    MaxThroughputWithLatencyBudgetStrategy,
    MinLatencyStrategy,
    RuntimeSearchStrategy,
    SelectedRuntimeStrategy,
)


@dataclasses.dataclass
class RuntimeAnalyzerResult:
    """Result for runtime analyzer.

    Args:
        latency: in milliseconds for selected result
        throughput: in samples/second for selected result
        model_status: details of selected model
        runner_status: details of selected runner
    """

    latency: float
    throughput: float
    model_status: ModelStatus
    runner_status: RunnerStatus


class RuntimeAnalyzer:
    """RuntimeAnalyzer class.

    Perform analysis of profiling results and obtain the best runtime and model based on
    the selected strategy and the status of pipeline execution.

    Example of use:

        RuntimeAnalyzer.get_runtime(
            models_status=models_status,
            strategy=MaxThroughputAndMinLatencyStrategy()
        )
    """

    @classmethod
    def get_runtime(
        cls,
        models_status: Dict[str, ModelStatus],
        strategy: RuntimeSearchStrategy,
        formats: Optional[Sequence[str]] = None,
        runners: Optional[Sequence[str]] = None,
    ) -> RuntimeAnalyzerResult:
        """Class method to obtain the best runtime and model for pipeline execution.

        Args:
            models_status: A statuses of generated and profiled models
            strategy: A strategy used to choose the best runtime
            formats: A list of formats that selection should be done from
            runners: A list of runners that selection should be done from

        Returns:
            A RuntimeAnalyzerResult object with model status and runner result for the best runtime

        Raises:
            ModelNavigatorRuntimeAnalyzerError: When no runtime satisfies the strategy requirements.
        """
        if isinstance(strategy, MinLatencyStrategy):
            result = cls._get_min_latency_runtime(
                models_status=models_status,
                formats=formats,
                runners=runners,
            )
        elif isinstance(strategy, MaxThroughputStrategy):
            result = cls._get_max_throughput_runtime(
                models_status=models_status,
                formats=formats,
                runners=runners,
            )
        elif isinstance(strategy, MaxThroughputAndMinLatencyStrategy):
            result = cls._get_max_throughput_runtime_min_latency_runtime(
                models_status=models_status,
                formats=formats,
                runners=runners,
            )
        elif isinstance(strategy, MaxThroughputWithLatencyBudgetStrategy):
            result = cls._get_max_throughput_runtime(
                models_status=models_status,
                formats=formats,
                runners=runners,
                latency_budget=strategy.latency_budget,
            )
        elif isinstance(strategy, SelectedRuntimeStrategy):
            result = cls._get_selected_runtime(
                models_status=models_status,
                model_key=strategy.model_key,
                runner_name=strategy.runner_name,
            )
        else:
            raise ModelNavigatorUserInputError(f"Unsupported strategy provided: {strategy}")

        if not result:
            raise ModelNavigatorRuntimeAnalyzerError("No matching results found.")

        LOGGER.info(
            f"\n"
            f"Strategy: {strategy}\n"
            f"  Latency: {result.latency:.4f} [ms]\n"
            f"  Throughput: {result.throughput:.4f} [infer/sec]\n"
            f"  Runner: {result.runner_status.runner_name}\n"
            f"  Model: {result.model_status.model_config.path.as_posix()}"
        )

        return result

    @classmethod
    def _get_min_latency_runtime(
        cls,
        *,
        models_status: Dict[str, ModelStatus],
        formats: Optional[Sequence[str]] = None,
        runners: Optional[Sequence[str]] = None,
    ) -> Optional[RuntimeAnalyzerResult]:
        best_latency, best_runtime = inf, None
        if formats is not None:
            models_status = {
                model_key: models_status
                for model_key, models_status in models_status.items()
                if models_status.model_config.format.value in formats
            }

        for model_status in models_status.values():
            runners_status = model_status.runners_status
            if runners is not None:
                runners_status = {
                    runner_name: runners_status
                    for runner_name, runners_status in runners_status.items()
                    if runners_status.runner_name in runners
                }

            for runner_status in runners_status.values():
                if (
                    runner_status.status.get(Correctness.__name__)
                    == runner_status.status.get(Performance.__name__)
                    == CommandStatus.OK
                ):
                    assert runner_status.result[Performance.__name__]["profiling_results"] is not None
                    latency = inf
                    throughput = None
                    for perf in runner_status.result[Performance.__name__]["profiling_results"]:
                        if perf.p50_latency < latency:
                            latency = perf.p50_latency
                            throughput = perf.throughput

                    if best_latency is None or latency < best_latency:
                        best_latency = latency
                        best_runtime = RuntimeAnalyzerResult(
                            latency=best_latency,
                            throughput=throughput,
                            model_status=model_status,
                            runner_status=runner_status,
                        )

        return best_runtime

    @classmethod
    def _get_max_throughput_runtime(
        cls,
        *,
        models_status: Dict[str, ModelStatus],
        latency_budget: Optional[float] = None,
        formats: Optional[Sequence[str]] = None,
        runners: Optional[Sequence[str]] = None,
    ) -> Optional[RuntimeAnalyzerResult]:
        best_throughput, best_runtime = -inf, None
        if formats is not None:
            models_status = {
                model_key: models_status
                for model_key, models_status in models_status.items()
                if models_status.model_config.format.value in formats
            }

        for model_status in models_status.values():
            runners_status = model_status.runners_status
            if runners is not None:
                runners_status = {
                    runner_name: runners_status
                    for runner_name, runners_status in runners_status.items()
                    if runners_status.runner_name in runners
                }

            for runner_status in runners_status.values():
                if (
                    runner_status.status.get(Correctness.__name__)
                    == runner_status.status.get(Performance.__name__)
                    == CommandStatus.OK
                ):
                    assert runner_status.result[Performance.__name__]["profiling_results"] is not None
                    latency = None
                    throughput = -inf
                    for perf in runner_status.result[Performance.__name__]["profiling_results"]:
                        if perf.throughput > throughput and (
                            latency_budget is None or perf.p50_latency <= latency_budget
                        ):
                            latency = perf.p50_latency
                            throughput = perf.throughput

                    if best_throughput is None or throughput > best_throughput:
                        best_throughput = throughput
                        best_runtime = RuntimeAnalyzerResult(
                            latency=latency,
                            throughput=throughput,
                            model_status=model_status,
                            runner_status=runner_status,
                        )

        return best_runtime

    @classmethod
    def _get_max_throughput_runtime_min_latency_runtime(
        cls,
        *,
        models_status: Dict[str, ModelStatus],
        formats: Optional[Sequence[str]] = None,
        runners: Optional[Sequence[str]] = None,
    ) -> RuntimeAnalyzerResult:
        min_lat_result = cls._get_min_latency_runtime(
            models_status=models_status,
            formats=formats,
            runners=runners,
        )
        max_thr_result = cls._get_max_throughput_runtime(
            models_status=models_status,
            formats=formats,
            runners=runners,
        )
        if (
            min_lat_result is not None
            and max_thr_result is not None
            and min_lat_result.model_status.model_config.path == max_thr_result.model_status.model_config.path
            and min_lat_result.runner_status.runner_name == max_thr_result.runner_status.runner_name
        ):
            return min_lat_result
        else:
            raise ModelNavigatorRuntimeAnalyzerError(
                "No runtime has both the minimal latency and the maximal throughput."
                "Consider using different `RuntimeSearchStrategy`"
            )

    @classmethod
    def _get_selected_runtime(
        cls,
        models_status: Dict[str, ModelStatus],
        model_key: str,
        runner_name: str,
    ):
        model_status = models_status.get(model_key)
        if not model_status:
            raise ModelNavigatorRuntimeAnalyzerError(f"Status for model {model_key} not found")

        runner_status = model_status.runners_status.get(runner_name)
        if not runner_status:
            raise ModelNavigatorRuntimeAnalyzerError(f"Status for model {model_key} and runner {runner_name} not found")

        profiling_results = runner_status.result[Performance.__name__]["profiling_results"]
        if len(profiling_results) == 0:
            raise ModelNavigatorRuntimeAnalyzerError(
                f"No profiling results for model {model_key} and runner {runner_name} not found"
            )

        perf = profiling_results[-1]

        result = RuntimeAnalyzerResult(
            latency=perf.p50_latency,
            throughput=perf.throughput,
            model_status=model_status,
            runner_status=runner_status,
        )
        return result
