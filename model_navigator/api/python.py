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

import pathlib
from typing import Callable, Optional, Sequence, Tuple, Type, Union

from model_navigator.api.config import (
    DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS,
    CustomConfig,
    DeviceKind,
    OptimizationProfile,
    SizedDataLoader,
    VerifyFunction,
    map_custom_configs,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config_builder import ModelConfigBuilder
from model_navigator.core.constants import DEFAULT_SAMPLE_COUNT
from model_navigator.frameworks import Framework
from model_navigator.package.package import Package
from model_navigator.pipelines.builders import (
    correctness_builder,
    performance_builder,
    preprocessing_builder,
    verify_builder,
)
from model_navigator.pipelines.wrappers.optimize import optimize_pipeline
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.utils import default_runners
from model_navigator.utils import enums


def optimize(
    model: Callable,
    dataloader: SizedDataLoader,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    batching: Optional[bool] = True,
    target_device: Optional[DeviceKind] = DeviceKind.CPU,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    optimization_profile: Optional[OptimizationProfile] = None,
    workspace: Optional[pathlib.Path] = None,
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
        runners: Use only runners provided as parameter
        optimization_profile: Optimization profile for conversion and profiling
        workspace: Workspace where packages will be extracted
        verbose: Enable verbose logging
        debug: Enable debug logging from commands
        verify_func: Function for additional model verification
        custom_configs: Sequence of CustomConfigs used to control produced artifacts

    Returns:
        Package descriptor representing created package.
    """
    if isinstance(model, str):
        model = pathlib.Path(model)

    if runners is None:
        runners = default_runners(device_kind=target_device)

    if optimization_profile is None:
        optimization_profile = OptimizationProfile()

    runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

    target_formats = DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS

    config = CommonConfig(
        Framework.NONE,
        model=model,
        dataloader=dataloader,
        target_formats=target_formats,
        target_device=target_device,
        sample_count=sample_count,
        batch_dim=0 if batching else None,
        runner_names=runner_names,
        optimization_profile=optimization_profile,
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

    builders = [
        preprocessing_builder,
        correctness_builder,
        performance_builder,
        verify_builder,
    ]

    package = optimize_pipeline(
        model=model,
        workspace=workspace,
        builders=builders,
        config=config,
        models_config=models_config,
    )

    return package
