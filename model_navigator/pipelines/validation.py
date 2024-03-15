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
"""Pipeline manager submodule."""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, get_args, get_origin

from model_navigator.api.config import (
    AVAILABLE_TARGET_FORMATS,
    CustomConfigForFormat,
    Format,
    OnnxConfig,
    ShapeTuple,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TensorRTProfile,
    TorchTensorRTConfig,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.exceptions import ModelNavigatorConfigurationError, ModelNavigatorConfigurationWarning
from model_navigator.frameworks import Framework
from model_navigator.package.package import Package
from model_navigator.runners.registry import get_runner
from model_navigator.utils.format_helpers import get_base_format, get_framework_export_formats


class PipelineManagerConfigurationValidator:
    """PipelineManager configuration validator."""

    @classmethod
    def run(
        cls,
        config: CommonConfig,
        package: Optional[Package] = None,
    ):
        """Validate PipelineManager configuration.

        Args:
            config: A configuration object
            package: Package to be optimized, if None package is yet to be built. Defaults to None.
        """
        cls._validate_if_runners_match_target_device(config)
        cls._validate_config_types(config)
        cls._validate_if_custom_configs_match_target_formats(config)
        cls._validate_if_target_formats_match_framework(config)
        cls._validate_optimization_profile_batch_sizes_when_batching_is_disabled(config)
        for custom_config in config.custom_configs.values():
            if isinstance(custom_config, (TensorRTConfig, TorchTensorRTConfig, TensorFlowTensorRTConfig)):
                cls._validate_trt_profile_input_names(custom_config.trt_profiles, config._input_names)
                for onnx_custom_config in config.custom_configs.values():
                    if isinstance(onnx_custom_config, OnnxConfig):
                        cls._validate_if_trt_profile_aligns_with_dynamic_axes(
                            onnx_custom_config.dynamic_axes, custom_config.trt_profiles
                        )
        if package is not None:
            cls._validate_if_target_formats_sources_are_available_in_package(
                package, config.target_formats, config.framework
            )

    @classmethod
    def _validate_if_runners_match_target_device(cls, config: CommonConfig):
        for runner_name in config.runner_names:
            runner = get_runner(runner_name)
            if config.target_device not in runner.devices_kind():
                raise ModelNavigatorConfigurationError(
                    f"Runner {runner_name} is configured for devices {runner.devices_kind()}, "
                    f"but target device is {config.target_device}."
                )

    @classmethod
    def _validate_if_custom_configs_match_target_formats(cls, config: CommonConfig):
        for custom_config in config.custom_configs.values():
            if isinstance(custom_config, CustomConfigForFormat) and custom_config.format not in config.target_formats:
                warnings.warn(
                    f"Custom configuration for format {custom_config.format} is provided, "
                    f"but {custom_config.format} is not in target formats. "
                    "Custom configuration will be ignored.",
                    ModelNavigatorConfigurationWarning,
                    stacklevel=1,
                )

    @classmethod
    def _validate_if_target_formats_match_framework(cls, config: CommonConfig):
        for target_format in config.target_formats:
            if target_format not in AVAILABLE_TARGET_FORMATS[config.framework]:
                raise ModelNavigatorConfigurationError(
                    f"{target_format} is not available for framework {config.framework}. "
                    f"Available target formats: ({AVAILABLE_TARGET_FORMATS[config.framework]})"
                )

    @classmethod
    def _validate_optimization_profile_batch_sizes_when_batching_is_disabled(cls, config: CommonConfig):
        if config.batch_dim is None and config.optimization_profile.batch_sizes:
            raise ModelNavigatorConfigurationError(
                f"Model does not support batching, but profiling batch sizes are {config.optimization_profile.batch_sizes}."
            )

    @classmethod
    def _validate_trt_profile_input_names(
        cls,
        trt_profiles: Optional[List[TensorRTProfile]],
        input_names: Optional[Tuple[str, ...]] = None,
    ):
        if input_names and trt_profiles:
            for trt_profile in trt_profiles:
                if not set(trt_profile) == set(input_names):
                    raise ModelNavigatorConfigurationError(
                        f"trt_profile input names: {trt_profile.keys()} are not "
                        f"matching model input names: {input_names}."
                    )

    @classmethod
    def _validate_if_target_formats_sources_are_available_in_package(
        cls, package: Package, target_formats: Sequence[Format], framework: Framework
    ):
        for target_format in target_formats:
            base_format = (
                target_format
                if target_format in get_framework_export_formats(framework)
                else get_base_format(target_format, framework)
            )
            base_format_available = False
            for model_status in package.status.models_status.values():
                if model_status.model_config.format == base_format and (
                    package.workspace.path / model_status.model_config.path
                ):
                    base_format_available = True
                    break
            if not base_format_available:
                warnings.warn(
                    f"Target format {target_format} requires {base_format} format "
                    "to be saved in the package but it is not found. "
                    "Target format will be skipped.",
                    ModelNavigatorConfigurationWarning,
                    stacklevel=1,
                )

    @classmethod
    def _validate_config_types(cls, config: CommonConfig) -> None:
        for field_name, field in config.__dataclass_fields__.items():
            expected_type = field.type
            if get_origin(expected_type) is Union:
                expected_type = tuple((get_origin(arg) or arg) for arg in get_args(expected_type))
            else:
                expected_type = get_origin(expected_type) or expected_type
            value = getattr(config, field_name)

            if isinstance(expected_type, (list, tuple)) and Any in expected_type:
                continue

            if not isinstance(value, expected_type):
                raise ModelNavigatorConfigurationError(
                    f"Incorrect type for {field_name}. Expected type {expected_type} got {type(value)} instead."
                )

        if config.from_source:
            try:
                iter(config.dataloader)
            except TypeError as e:
                raise ModelNavigatorConfigurationError("Datalaoder must be iterable.") from e
            try:
                len(config.dataloader)
            except TypeError as e:
                raise ModelNavigatorConfigurationError("Datalaoder must must have len().") from e

    @classmethod
    def _validate_if_trt_profile_aligns_with_dynamic_axes(
        cls,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]],
        trt_profiles: Optional[List[TensorRTProfile]],
    ):
        def _get_trt_dynamic_axes(shape_tuple: ShapeTuple) -> List[int]:
            return [idx for idx, min_shape in enumerate(shape_tuple.min) if min_shape != shape_tuple.max[idx]]

        if dynamic_axes is None or trt_profiles is None:
            return
        for trt_profile in trt_profiles:
            for input_name, shape_tuple in trt_profile.items():
                for trt_dynamic_ax in _get_trt_dynamic_axes(shape_tuple):
                    if trt_dynamic_ax not in dynamic_axes[input_name]:
                        raise ModelNavigatorConfigurationError(
                            f"Dynamic axis {trt_dynamic_ax} is not specified in dynamic_axes for input {input_name}."
                        )
