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
"""TensorRT optimize API."""

from pathlib import Path
from typing import Optional, Tuple, Type, Union

from model_navigator.api.config import (
    DEFAULT_TENSORRT_TARGET_FORMATS,
    DeviceKind,
    OptimizationProfile,
    SizedDataLoader,
    VerifyFunction,
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
from model_navigator.pipelines.builders.find_device_max_batch_size import find_device_max_batch_size_builder
from model_navigator.pipelines.wrappers.optimize import optimize_pipeline
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.utils import default_runners
from model_navigator.utils import enums


def optimize(
    model: Union[Path, str],
    dataloader: SizedDataLoader,
    sample_count: Optional[int] = DEFAULT_SAMPLE_COUNT,
    batching: Optional[bool] = True,
    runners: Optional[Union[Union[str, Type[NavigatorRunner]], Tuple[Union[str, Type[NavigatorRunner]], ...]]] = None,
    optimization_profile: Optional[OptimizationProfile] = None,
    workspace: Optional[Path] = None,
    verbose: bool = False,
    debug: bool = False,
    verify_func: Optional[VerifyFunction] = None,
) -> Package:
    """Function executes correctness test, performance profiling and optional verification on provided TensorRT model.

    Args:
        model: TensorRT model path or string
        dataloader: Sized iterable with data that will be feed to the model
        sample_count: Limits how many samples will be used from dataloader
        batching: Enable or disable batching on first (index 0) dimension of the model
        runners: Use only runners provided as parameter
        optimization_profile: Optimization profile for conversion and profiling
        workspace: Workspace where packages will be extracted
        verbose: Enable verbose logging
        debug: Enable debug logging from commands
        verify_func: Function for additional model verification

    Returns:
        Package descriptor representing created package.
    """
    if isinstance(model, str):
        model = Path(model)
    target_formats = DEFAULT_TENSORRT_TARGET_FORMATS
    target_device = DeviceKind.CUDA

    if runners is None:
        runners = default_runners(device_kind=target_device)

    if optimization_profile is None:
        optimization_profile = OptimizationProfile()

    runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

    config = CommonConfig(
        Framework.TENSORRT,
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
    )

    models_config = ModelConfigBuilder.generate_model_config(
        framework=Framework.TENSORRT,
        target_formats=target_formats,
        custom_configs=None,
    )

    builders = [
        preprocessing_builder,
        find_device_max_batch_size_builder,
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
