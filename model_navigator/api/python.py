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
"""Python optimize API."""
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, Type, Union

from model_navigator.api.config import (
    DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS,
    CustomConfig,
    DeviceKind,
    ProfilerConfig,
    SizedDataLoader,
    VerifyFunction,
    map_custom_configs,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config_builder import ModelConfigBuilder
from model_navigator.core.constants import DEFAULT_SAMPLE_COUNT
from model_navigator.core.package import Package
from model_navigator.frameworks import Framework
from model_navigator.pipelines.builders import (
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
    verify_builder,
)
from model_navigator.pipelines.pipeline_manager import PipelineManager
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.utils import default_runners
from model_navigator.utils import enums
from model_navigator.utils.common import get_default_workspace


def optimize(
    model: Callable,
    dataloader: SizedDataLoader,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    batching: Optional[bool] = True,
    target_device: Optional[DeviceKind] = DeviceKind.CPU,
    runners: Optional[Union[Union[str, Type[NavigatorRunner]], Tuple[Union[str, Type[NavigatorRunner]], ...]]] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    workspace: Optional[Path] = None,
    verbose: bool = False,
    debug: bool = False,
    verify_func: Optional[VerifyFunction] = None,
    custom_configs: Optional[Sequence[CustomConfig]] = None,
) -> Package:
    """Entrypoint for Python model optimize.

    Perform correctness testing, profiling and model verification.

    Args:
        model: Model inference function
        dataloader: Sized iterable with data that will be feed to the model
        sample_count: Limits how many samples will be used from dataloader
        batching: Enable or disable batching on first (index 0) dimension of the model
        target_device: Target device for optimize process, default is CPU
        runners: Use only runners provided as paramter
        profiler_config: Profiling config
        workspace: Workspace where packages will be extracted
        verbose: Enable verbose logging
        debug: Enable debug logging from commands
        verify_func: Function for additional model verifcation
        custom_configs: Sequence of CustomConfigs used to control produced artifacts

    Returns:
        Package descriptor representing created package.
    """
    if isinstance(model, str):
        model = Path(model)
    if workspace is None:
        workspace = get_default_workspace()

    sample = next(iter(dataloader))
    forward_kw_names = tuple(sample.keys()) if isinstance(sample, Mapping) else None

    if runners is None:
        runners = default_runners(device_kind=target_device)

    if profiler_config is None:
        profiler_config = ProfilerConfig()

    runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

    target_formats = DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS
    config = CommonConfig(
        Framework.NONE,
        model=model,
        dataloader=dataloader,
        forward_kw_names=forward_kw_names,
        workspace=workspace,
        target_formats=target_formats,
        target_device=target_device,
        sample_count=sample_count,
        batch_dim=0 if batching else None,
        runner_names=runner_names,
        profiler_config=profiler_config,
        verbose=verbose,
        debug=debug,
        verify_func=verify_func,
        custom_configs=map_custom_configs(custom_configs=custom_configs),
    )

    models_config = ModelConfigBuilder.generate_model_config(
        framework=Framework.NONE,
        target_formats=target_formats,
        custom_configs=[],
    )

    builders = [preprocessing_builder, correctness_builder]
    if profiler_config.run_profiling:
        builders.append(profiling_builder)
    builders.append(verify_builder)

    package = PipelineManager.run(
        pipeline_builders=builders,
        config=config,
        models_config=models_config,
    )

    return package
