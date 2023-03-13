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
"""Torch optimize API."""
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Type, Union

import torch  # pytype: disable=import-error

from model_navigator.api.config import (
    DEFAULT_TORCH_TARGET_FORMATS,
    CustomConfig,
    DeviceKind,
    Format,
    ProfilerConfig,
    SizedDataLoader,
    VerifyFunction,
    map_custom_configs,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config_builder import ModelConfigBuilder
from model_navigator.constants import DEFAULT_SAMPLE_COUNT
from model_navigator.core.package import Package
from model_navigator.logger import LOGGER
from model_navigator.pipelines.builders import (
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
    torch_conversion_builder,
    torch_export_builder,
    verify_builder,
)
from model_navigator.pipelines.pipeline_manager import PipelineManager
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.utils import default_runners
from model_navigator.utils import enums
from model_navigator.utils.common import get_default_workspace
from model_navigator.utils.framework import Framework
from model_navigator.utils.torch import update_allowed_batching_parameters


def optimize(
    model: torch.nn.Module,
    dataloader: SizedDataLoader,
    sample_count: Optional[int] = DEFAULT_SAMPLE_COUNT,
    batching: Optional[bool] = True,
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    target_formats: Optional[Union[Union[str, Format], Tuple[Union[str, Format], ...]]] = None,
    target_device: Optional[DeviceKind] = DeviceKind.CUDA,
    runners: Optional[Union[Union[str, Type[NavigatorRunner]], Tuple[Union[str, Type[NavigatorRunner]], ...]]] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    workspace: Optional[Path] = None,
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    verify_func: Optional[VerifyFunction] = None,
    custom_configs: Optional[Sequence[CustomConfig]] = None,
) -> Package:
    """Function exports PyTorch model to all supported formats.

    Args:
        model: PyTorch model object
        dataloader: Sized iterable with data that will be feed to the model
        sample_count: Limits how many samples will be used from dataloader
        batching: Enable or disable batching on first (index 0) dimension of the model
        input_names: Model input names
        output_names: Model output names
        target_formats: Target model formats for optimize process
        target_device: Target device for optimize process, default is CUDA
        runners: Use only runners provided as parameter
        profiler_config: Profiling config
        workspace: Workspace where packages will be extracted
        verbose: Enable verbose logging
        debug: Enable debug logging from commands
        verify_func: Function for additional model verification
        custom_configs: Sequence of CustomConfigs used to control produced artifacts

    Returns:
        Package descriptor representing created package.
    """
    if workspace is None:
        workspace = get_default_workspace()
    if target_formats is None:
        target_formats = DEFAULT_TORCH_TARGET_FORMATS
        if batching:
            target_formats, custom_configs = update_allowed_batching_parameters(
                target_formats=target_formats,
                custom_configs=custom_configs,
            )
        LOGGER.info(f"Using default target formats: {[tf.name for tf in target_formats]}")

    sample = next(iter(dataloader))
    if isinstance(sample, Mapping):
        forward_kw_names = tuple(sample.keys())
    else:
        forward_kw_names = None

    if runners is None:
        runners = default_runners(device_kind=target_device)

    target_formats = enums.parse(target_formats, Format)
    runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

    if profiler_config is None:
        profiler_config = ProfilerConfig()

    if Format.TORCH not in target_formats:
        target_formats = (Format.TORCH,) + target_formats

    config = CommonConfig(
        framework=Framework.TORCH,
        model=model,
        dataloader=dataloader,
        target_formats=target_formats,
        workspace=workspace,
        sample_count=sample_count,
        _input_names=input_names,
        _output_names=output_names,
        target_device=target_device,
        forward_kw_names=forward_kw_names,
        batch_dim=0 if batching else None,
        runner_names=runner_names,
        profiler_config=profiler_config,
        verbose=verbose,
        debug=debug,
        verify_func=verify_func,
        custom_configs=map_custom_configs(custom_configs=custom_configs),
    )

    models_config = ModelConfigBuilder.generate_model_config(
        framework=Framework.TORCH,
        target_formats=target_formats,
        custom_configs=custom_configs,
    )

    builders = [
        preprocessing_builder,
        torch_export_builder,
        torch_conversion_builder,
        correctness_builder,
    ]
    if profiler_config.run_profiling:
        builders.append(profiling_builder)
    builders.append(verify_builder)

    package = PipelineManager.run(
        pipeline_builders=builders,
        config=config,
        models_config=models_config,
    )

    return package
