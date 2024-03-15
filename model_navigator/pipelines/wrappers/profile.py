# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Execute package profiling pipelines and return profiling results."""

from typing import Dict, List, Optional, Sequence

from model_navigator.api.config import Format
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.performance import Profile
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model import model_config
from model_navigator.package.package import Package
from model_navigator.package.profiling_results import (
    ProfilingResult,
    ProfilingResults,
    RunnerProfilingResults,
    RunnerResults,
)
from model_navigator.pipelines.builders import PipelineBuilder
from model_navigator.pipelines.pipeline_context import PipelineContext
from model_navigator.pipelines.pipeline_manager import PipelineManager


def profile_pipeline(
    package: Package,
    builders: Sequence[PipelineBuilder],
    config: CommonConfig,
    models_config: Optional[Dict[Format, List[model_config.ModelConfig]]] = None,
) -> ProfilingResults:
    """Execute package profiling for provided dataloader and return profiling results.

    Args:
        builders: Pipeline builders
        config: Common configuration
        package: Package to optimize
        models_config: Model configs used during optimize

    Returns:
        Profiling results
    """
    pipeline_manager = PipelineManager(workspace=package.workspace)
    context = pipeline_manager.run(
        workspace=package.workspace,
        builders=builders,
        config=config,
        models_config=models_config,
        package=package,
    )

    builder = ProfilingResultsBuilder()
    profiling_results = builder.run(context=context)

    return profiling_results


class ProfilingResultsBuilder:
    """Profiling results builder."""

    def run(
        self,
        context: PipelineContext,
    ) -> ProfilingResults:
        """Create results from pipeline context.

        Args:
            context: Pipeline context with execution data

        Returns:
            Profiling results
        """
        models_results = {}
        samples_data = {}
        for model_key, model_command in context.commands.models_commands.items():
            runners_results = {}
            for runner_key, runner_command in model_command.runners_commands.items():
                command = runner_command.commands.get(Profile.name)
                if not command:
                    continue

                if command.status != CommandStatus.OK:
                    runners_results[runner_key] = RunnerProfilingResults(
                        status=command.status,
                        detailed={},
                    )
                    continue

                profiling_results = command.output["profiling_results"]
                profiling_samples = command.output["profiling_samples"]
                if len(samples_data) < len(profiling_samples):
                    samples_data = dict(enumerate(profiling_samples))

                detailed = {}
                for result in profiling_results:
                    profiling_result = ProfilingResult(
                        batch_size=result.batch_size,
                        avg_latency=result.avg_latency,
                        std_latency=result.std_latency,
                        p50_latency=result.p50_latency,
                        p90_latency=result.p90_latency,
                        p95_latency=result.p95_latency,
                        p99_latency=result.p99_latency,
                        throughput=result.throughput,
                        avg_gpu_clock=result.avg_gpu_clock,
                        request_count=result.request_count,
                    )
                    res = detailed.get(result.sample_id, [])
                    res.append(profiling_result)
                    detailed[result.sample_id] = res

                runners_results[runner_key] = RunnerProfilingResults(
                    status=command.status,
                    detailed=detailed,
                )

            models_results[model_key] = RunnerResults(runners=runners_results)

        profiling_results = ProfilingResults(
            models=models_results,
            samples_data=samples_data,
        )

        return profiling_results
