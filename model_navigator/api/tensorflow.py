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
"""TensorFlow optimize API."""
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Type, Union

import tensorflow  # pytype: disable=import-error

from model_navigator.api.config import (
    DEFAULT_TENSORFLOW_TARGET_FORMATS,
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
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.pipelines.builders import (
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
    tensorflow_conversion_builder,
    tensorflow_export_builder,
    verify_builder,
)
from model_navigator.pipelines.pipeline_manager import PipelineManager
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.utils import default_runners
from model_navigator.utils import enums
from model_navigator.utils.common import get_default_workspace
from model_navigator.utils.framework import Framework


def optimize(
    model: tensorflow.keras.Model,
    dataloader: SizedDataLoader,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    batching: Optional[bool] = True,
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    target_formats: Optional[Union[Union[str, Format], Tuple[Union[str, Format], ...]]] = None,
    target_device: Optional[DeviceKind] = DeviceKind.CUDA,
    runners: Optional[Union[Union[str, Type[NavigatorRunner]], Tuple[Union[str, Type[NavigatorRunner]], ...]]] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    workspace: Optional[Path] = None,
    verbose: bool = False,
    debug: bool = False,
    verify_func: Optional[VerifyFunction] = None,
    custom_configs: Optional[Sequence[CustomConfig]] = None,
) -> Package:
    """Function exports TensorFlow2 model to all supported formats.

    Args:
        model: TensorFlow2 model object
        dataloader: Sized iterable with data that will be feed to the model
        sample_count: Limits how many samples will be used from dataloader
        batching: Enable or disable batching on first (index 0) dimension of the model
        input_names: Model input names
        output_names: Model output names
        target_formats: Target model formats for optimize process
        target_device: Target device for optimize process, default is CUDA
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
    if target_device == DeviceKind.CPU and any(
        [device.device_type == "GPU" for device in tensorflow.config.get_visible_devices()]
    ):
        raise ModelNavigatorConfigurationError(
            "\n"
            "    'target_device == nav.DeviceKind.CPU' is not supported for TensorFlow2 when GPU is available.\n"
            "    To optimize model for CPU, disable GPU with: "
            "'tf.config.set_visible_devices([], 'GPU')' directly after importing TensorFlow.\n"
        )
    if workspace is None:
        workspace = get_default_workspace()
    if target_formats is None:
        target_formats = DEFAULT_TENSORFLOW_TARGET_FORMATS

    if runners is None:
        runners = default_runners(device_kind=target_device)

    forward_kw_names = None
    sample = next(iter(dataloader))
    if isinstance(sample, Mapping):
        forward_kw_names = tuple(sample.keys())

    target_formats_enums = enums.parse(target_formats, Format)
    runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

    if profiler_config is None:
        profiler_config = ProfilerConfig()

    if Format.TENSORFLOW not in target_formats_enums:
        target_formats_enums = (Format.TENSORFLOW,) + target_formats_enums

    config = CommonConfig(
        Framework.TENSORFLOW,
        model=model,
        dataloader=dataloader,
        target_formats=target_formats_enums,
        target_device=target_device,
        workspace=workspace,
        sample_count=sample_count,
        _input_names=input_names,
        _output_names=output_names,
        batch_dim=0 if batching else None,
        runner_names=runner_names,
        profiler_config=profiler_config,
        forward_kw_names=forward_kw_names,
        verbose=verbose,
        debug=debug,
        verify_func=verify_func,
        custom_configs=map_custom_configs(custom_configs=custom_configs),
    )

    models_config = ModelConfigBuilder.generate_model_config(
        framework=Framework.TENSORFLOW,
        target_formats=target_formats_enums,
        custom_configs=custom_configs,
    )

    builders = [
        preprocessing_builder,
        tensorflow_export_builder,
        tensorflow_conversion_builder,
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
