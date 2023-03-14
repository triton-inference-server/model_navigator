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
"""Package operations related API."""


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

from model_navigator.api.config import (
    DEFAULT_TARGET_FORMATS,
    SOURCE_FORMATS,
    CustomConfig,
    DeviceKind,
    Format,
    ProfilerConfig,
    VerifyFunction,
    map_custom_configs,
)
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.verification.verify import VerifyModel
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.configuration.model.model_config_builder import ModelConfigBuilder
from model_navigator.core.package import Package
from model_navigator.core.status import ModelStatus
from model_navigator.exceptions import (
    ModelNavigatorEmptyPackageError,
    ModelNavigatorMissingSourceModelError,
    ModelNavigatorNotFoundError,
)
from model_navigator.pipelines.builders import (
    PipelineBuilder,
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
)
from model_navigator.pipelines.builders.find_device_max_batch_size import find_device_max_batch_size_builder
from model_navigator.pipelines.builders.verify import verify_builder
from model_navigator.pipelines.pipeline_manager import PipelineManager
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import runner_registry
from model_navigator.runners.utils import default_runners
from model_navigator.runtime_analyzer.strategy import RuntimeSearchStrategy
from model_navigator.utils import enums
from model_navigator.utils.format_helpers import FRAMEWORK2BASE_FORMAT
from model_navigator.utils.framework import Framework
from model_navigator.utils.torch import update_allowed_batching_parameters


def load(
    path: Union[str, Path],
    workspace: Optional[Union[str, Path]] = None,
) -> Package:
    """Load package from provided path.

    Args:
        path: The location of package to load
        workspace: Workspace where packages will be extracted

    Returns:
        Package.
    """
    return Package.load(path=path, workspace=workspace)


def save(
    package: Package,
    path: Union[str, Path],
    keep_workspace: bool = True,
    override: bool = False,
    save_data: bool = True,
) -> None:
    """Save export results into the .nav package at given path.

    Args:
        package: A package object to prepare the package
        path: A path to file where the package has to be saved
        keep_workspace: flag to remove the working directory after saving the package
        override: flag to override existing package in provided path
        save_data: disable saving samples from the dataloader
    """
    package.save(
        path=path,
        keep_workspace=keep_workspace,
        override=override,
        save_data=save_data,
    )


def get_best_model_status(
    package: Package,
    strategy: Optional[RuntimeSearchStrategy] = None,
    include_source: bool = True,
) -> Optional[ModelStatus]:
    """Returns ModelStatus of best model for given strategy.

    If model with given strategy cannot be found, search is repeated with MaxThroughputStrategy.
    If there is no model match given strategy or MaxThroughputStrategy, function returns None.

    Args:
        package: A package object to be searched for best model.
        strategy: Strategy for finding the best model. Defaults to `MaxThroughputAndMinLatencyStrategy`
        include_source: Flag if Python based model has to be included in analysis

    Returns:
        ModelStatus of best model for given strategy or None.
    """
    return package.get_best_model_status(strategy=strategy, include_source=include_source)


def optimize(
    package: Package,
    target_formats: Optional[Union[Union[str, Format], Tuple[Union[str, Format], ...]]] = None,
    target_device: Optional[DeviceKind] = DeviceKind.CUDA,
    runners: Optional[Union[Union[str, Type[NavigatorRunner]], Tuple[Union[str, Type[NavigatorRunner]], ...]]] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    verbose: bool = False,
    debug: bool = False,
    verify_func: Optional[VerifyFunction] = None,
    custom_configs: Optional[List[CustomConfig]] = None,
    defaults: bool = True,
) -> Package:
    """Generate target formats and run correctness and profiling tests for available runners.

    Args:
        package: Package to optimize.
        target_formats: Formats to generate and profile. Defaults to target formats from the package.
        target_device: Target device for optimize process, default is CUDA
        runners: Runners to run correctness tests and profiling on. Defaults to runners from the package.
        profiler_config: Configuration of the profiler. Defaults to config from the package.
        verbose: If True enable verbose logging. Defaults to False.
        debug: If True print debugging logs. Defaults to False.
        verify_func: Function used for verifying generated models. Defaults to None.
        custom_configs: Custom formats configuration. Defaults to None.
        defaults: reset configuration of custom configs to defaults

    Returns:
        Optimized package
    """
    if package.is_empty() and package.model is None:
        raise ModelNavigatorEmptyPackageError(
            "Package is empty and source model is not loaded. Unable to run optimize."
        )
    config = package.config

    if target_formats is None:
        target_formats = DEFAULT_TARGET_FORMATS[package.framework]
        if package.framework == Framework.TORCH and config.batch_dim is not None:
            target_formats, custom_configs = update_allowed_batching_parameters(
                target_formats=target_formats,
                custom_configs=custom_configs,
            )

    if runners is None:
        runners = default_runners(device_kind=target_device)

    if profiler_config is None:
        profiler_config = ProfilerConfig()

    is_source_available = package.model is not None
    _update_config(
        config=config,
        is_source_available=is_source_available,
        target_formats=target_formats,
        runners=runners,
        profiler_config=profiler_config,
        verbose=verbose,
        debug=debug,
        verify_func=verify_func,
        custom_configs=custom_configs,
        defaults=defaults,
        target_device=target_device,
    )

    builders = _get_builders(
        framework=package.framework,
        run_profiling=config.profiler_config.run_profiling,
    )

    model_configs = _get_model_configs(
        config=config,
        custom_configs=list(config.custom_configs.values()),
    )

    optimized_package = PipelineManager.run(
        pipeline_builders=builders,
        config=config,
        models_config=model_configs,
        package=package,
    )

    return optimized_package


def set_verified(
    package: Package,
    model_key: str,
    runner_name: str,
) -> None:
    """Set verified status for model and runner.

    Args:
        package (Package): Package.
        model_key (str): Unique key of the model.
        runner_name (str): Name of the runner.

    Raises:
        ModelNavigatorNotFoundError: When model and runner not found.
    """
    try:
        runner_results = package.status.models_status[model_key].runners_status[runner_name]
    except KeyError:
        raise ModelNavigatorNotFoundError(f"Model {model_key} and runner {runner_name} not found.")
    runner_results.status[VerifyModel.__name__] = CommandStatus.OK


def _get_builders(framework: Framework, run_profiling: bool) -> List[PipelineBuilder]:
    """Build list of pipeline builders for nav.package.optimize.

    Args:
        framework (Framework): Package framework.
        run_profiling (bool): If True attach profiling pipeline builder.
    """
    if framework == Framework.TORCH:
        from model_navigator.pipelines.builders import torch_conversion_builder as conversion_builder
    elif framework in (Framework.TENSORFLOW, Framework.JAX):
        from model_navigator.pipelines.builders import tensorflow_conversion_builder as conversion_builder
    else:
        assert framework == Framework.ONNX
        from model_navigator.pipelines.builders import onnx_conversion_builder as conversion_builder

    builders: List[PipelineBuilder] = [
        preprocessing_builder,
        find_device_max_batch_size_builder,
        conversion_builder,
        correctness_builder,
    ]
    if run_profiling:
        builders.append(profiling_builder)
    builders.append(verify_builder)
    return builders


def _get_model_configs(config: CommonConfig, custom_configs: List[CustomConfig]) -> Dict[Format, List[ModelConfig]]:
    """Build model configs for nav.package.optimize.

    Args:
        config: Common configuration.
        custom_configs: List of custom configs.
    """
    custom_model_configs = ModelConfigBuilder.generate_model_config(
        framework=config.framework,
        target_formats=config.target_formats,
        custom_configs=custom_configs,
    )
    return custom_model_configs


def _update_config(
    config: CommonConfig,
    is_source_available: bool,
    target_formats: Optional[Union[Union[str, Format], Tuple[Union[str, Format], ...]]],
    *,
    runners: Optional[Union[Union[str, Type[NavigatorRunner]], Tuple[Union[str, Type[NavigatorRunner]], ...]]] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    verbose: bool = False,
    debug: bool = False,
    verify_func: Optional[VerifyFunction] = None,
    custom_configs: Optional[List[CustomConfig]] = None,
    defaults: bool = True,
    target_device: DeviceKind = DeviceKind.CUDA,
) -> None:
    base_format = FRAMEWORK2BASE_FORMAT[config.framework]

    # Reset target formats
    target_formats_enums = enums.parse(target_formats, Format)
    if base_format in target_formats_enums and base_format in SOURCE_FORMATS and not is_source_available:
        raise ModelNavigatorMissingSourceModelError(
            "Source model is not available in the package.\n"
            "Load source model with package.load_source_model(model) to use it "
            f"or remove {base_format} from target_formats."
        )
    config.target_formats = target_formats_enums

    # Reset profiling config
    if profiler_config is None:
        profiler_config = ProfilerConfig()

    config.profiler_config = profiler_config

    # Reset runner names
    if runners is None:
        runners = tuple(runner_registry.keys())

    config.runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

    # Update verify function
    if verify_func is not None:
        config.verify_func = verify_func

    # Reset custom config to defaults
    if defaults:
        for custom_config in config.custom_configs.values():
            custom_config.defaults()

    if custom_configs is not None:
        mapped_custom_configs = map_custom_configs(custom_configs=custom_configs)
        config.custom_configs.update(**mapped_custom_configs)

    config.target_device = target_device
    config.verbose = verbose
    config.debug = debug
    config.from_source = False
