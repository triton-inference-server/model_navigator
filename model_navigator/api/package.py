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

import pathlib
from typing import Dict, List, Optional, Tuple, Type, Union

from model_navigator.api.config import (
    SOURCE_FORMATS,
    CustomConfig,
    DeviceKind,
    Format,
    OptimizationProfile,
    SizedDataLoader,
    VerifyFunction,
    map_custom_configs,
)
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.verification.verify import VerifyModel
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.configuration.model.model_config_builder import ModelConfigBuilder
from model_navigator.core.constants import DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD
from model_navigator.core.logger import LOGGER
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import (
    ModelNavigatorEmptyPackageError,
    ModelNavigatorMissingSourceModelError,
    ModelNavigatorNotFoundError,
)
from model_navigator.frameworks import Framework
from model_navigator.frameworks.torch.utils import update_allowed_batching_parameters
from model_navigator.package.builder import PackageBuilder
from model_navigator.package.loader import PackageLoader
from model_navigator.package.package import Package
from model_navigator.package.status import ModelStatus
from model_navigator.pipelines.builders import (
    PipelineBuilder,
    correctness_builder,
    performance_builder,
    preprocessing_builder,
)
from model_navigator.pipelines.builders.find_device_max_batch_size import find_device_max_batch_size_builder
from model_navigator.pipelines.builders.profiling import profiling_builder
from model_navigator.pipelines.builders.verify import verify_builder
from model_navigator.pipelines.wrappers.optimize import optimize_pipeline
from model_navigator.pipelines.wrappers.profile import ProfilingResults, profile_pipeline
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import runner_registry
from model_navigator.runners.utils import default_runners
from model_navigator.runtime_analyzer.strategy import RuntimeSearchStrategy
from model_navigator.utils import enums
from model_navigator.utils.format_helpers import FRAMEWORK2BASE_FORMAT, get_target_formats


def load(
    path: Union[str, pathlib.Path],
    workspace: Optional[Union[str, pathlib.Path]] = None,
) -> Package:
    """Load package from provided path.

    Args:
        path: The location of package to load
        workspace: Workspace where packages will be extracted

    Returns:
        Package.
    """
    LOGGER.info(f"Loading package from {path} to {workspace}.")
    workspace = Workspace(workspace)
    workspace.initialize()

    loader = PackageLoader()
    package = loader.from_file(path=path, workspace=workspace)
    LOGGER.info(f"Package loaded and unpacked {workspace}.")

    return package


def load_from_workspace(
    workspace: Optional[Union[str, pathlib.Path]] = None,
) -> Package:
    """Load package from provided workspace.

    Args:
        workspace: The location of workspace to load

    Returns:
        Package.
    """
    LOGGER.info(f"Loading package from {workspace}.")

    loader = PackageLoader()
    package = loader.from_workspace(workspace)
    LOGGER.info(f"Package loaded and unpacked {workspace}.")

    return package


def save(
    package: Package,
    path: Union[str, pathlib.Path],
    override: bool = False,
    save_data: bool = True,
) -> None:
    """Save export results into the .nav package at given path.

    Args:
        package: A package object to prepare the package
        path: A path to file where the package has to be saved
        override: flag to override existing package in provided path
        save_data: disable saving samples from the dataloader
    """
    builder = PackageBuilder()
    builder.save(
        package=package,
        path=path,
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
    target_formats: Optional[Tuple[Union[str, Format], ...]] = None,
    target_device: Optional[DeviceKind] = DeviceKind.CUDA,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    optimization_profile: Optional[OptimizationProfile] = None,
    verbose: bool = False,
    debug: bool = False,
    verify_func: Optional[VerifyFunction] = None,
    custom_configs: Optional[List[CustomConfig]] = None,
    defaults: bool = True,
    fail_on_empty: bool = True,
) -> Package:
    """Generate target formats and run correctness and profiling tests for available runners.

    Args:
        package: Package to optimize.
        target_formats: Formats to generate and profile. Defaults to target formats from the package.
        target_device: Target device for optimize process, default is CUDA
        runners: Runners to run correctness tests and profiling on. Defaults to runners from the package.
        optimization_profile: Optimization profile used for conversion and profiling.
        verbose: If True enable verbose logging. Defaults to False.
        debug: If True print debugging logs. Defaults to False.
        verify_func: Function used for verifying generated models. Defaults to None.
        custom_configs: Custom formats configuration. Defaults to None.
        defaults: reset configuration of custom configs to defaults
        fail_on_empty: Fail optimization when empty (no model or base exported model) package provided

    Returns:
        Optimized package
    """
    if fail_on_empty and package.is_empty() and package.model is None:
        raise ModelNavigatorEmptyPackageError(
            "Package is empty and source model is not loaded. Unable to run optimize."
        )
    config = package.config

    is_source_available = package.model is not None
    if target_formats is None:
        target_formats = get_target_formats(framework=package.framework, is_source_available=is_source_available)
        if package.framework == Framework.TORCH and config.batch_dim is not None:
            target_formats, custom_configs = update_allowed_batching_parameters(
                target_formats=target_formats,
                custom_configs=custom_configs,
            )

    if runners is None:
        runners = default_runners(device_kind=target_device)

    if optimization_profile is None:
        optimization_profile = OptimizationProfile()

    _update_config(
        config=config,
        is_source_available=is_source_available,
        target_formats=target_formats,
        runners=runners,
        optimization_profile=optimization_profile,
        verbose=verbose,
        debug=debug,
        verify_func=verify_func,
        custom_configs=custom_configs,
        defaults=defaults,
        target_device=target_device,
    )

    builders = _get_builders(
        framework=package.framework,
    )

    models_config = _get_model_configs(
        config=config,
        custom_configs=list(config.custom_configs.values()),
    )

    optimized_package = optimize_pipeline(
        package=package,
        workspace=package.workspace.path,
        builders=builders,
        config=config,
        models_config=models_config,
    )

    return optimized_package


def profile(
    package: Package,
    dataloader: Optional[SizedDataLoader] = None,
    target_formats: Optional[Tuple[Union[str, Format], ...]] = None,
    target_device: Optional[DeviceKind] = DeviceKind.CUDA,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    max_batch_size: Optional[int] = None,
    batch_sizes: Optional[List[int]] = None,
    window_size: int = 50,
    stability_percentage: float = 10.0,
    stabilization_windows: int = 3,
    min_trials: int = 3,
    max_trials: int = 10,
    throughput_cutoff_threshold: float = DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD,
    verbose: bool = False,
) -> ProfilingResults:
    """Profile provided package.

    When `dataloader` is provided, use all samples obtained from dataloader per-each batch size to perform profiling.
    The profiling result return the min, max and average results per batch size from all samples.

    When no `dataloader` provided, the profiling sample from package is used.

    Args:
        package: Package to profile.
        dataloader: Sized iterable with data that will be feed to the model
        target_formats: Formats to profile. Defaults to target formats from the package.
        target_device: Target device to run profiling on.
        runners: Runners to run profiling on. Defaults to runners from the package.
        max_batch_size: Maximal batch size used for profiling. Default: None
        batch_sizes: List of batch sizes to profile. Default: None
        window_size: Number of inference queries performed in measurement window
        stability_percentage: Allowed percentage of variation from the mean in three consecutive windows.
        stabilization_windows: Number consecutive windows selected for stabilization.
        min_trials: Minimal number of window trials.
        max_trials: Maximum number of window trials.
        throughput_cutoff_threshold: Minimum throughput increase to continue profiling.
        verbose: If True enable verbose logging. Defaults to False.

    Returns:
        Profiling results
    """
    if package.is_empty() and package.model is None:
        raise ModelNavigatorEmptyPackageError(
            "Package is empty and source model is not loaded. Unable to run optimize."
        )

    config = package.config
    is_source_available = package.model is not None

    if target_formats is None:
        target_formats = get_target_formats(framework=package.framework, is_source_available=is_source_available)
        if package.framework == Framework.TORCH and config.batch_dim is not None:
            target_formats, _custom_configs = update_allowed_batching_parameters(
                target_formats=target_formats,
                custom_configs=(),
            )

    if runners is None:
        runners = default_runners(device_kind=target_device)

    if dataloader is None:
        dataloader = []

    optimization_profile = OptimizationProfile(
        max_batch_size=max_batch_size,
        batch_sizes=batch_sizes,
        window_size=window_size,
        stability_percentage=stability_percentage,
        stabilization_windows=stabilization_windows,
        min_trials=min_trials,
        max_trials=max_trials,
        throughput_cutoff_threshold=throughput_cutoff_threshold,
    )

    _update_config(
        config=config,
        dataloader=dataloader,
        is_source_available=is_source_available,
        target_formats=target_formats,
        runners=runners,
        optimization_profile=optimization_profile,
        verbose=verbose,
        target_device=target_device,
    )

    builders = [
        preprocessing_builder,
        profiling_builder,
    ]

    model_configs = _get_model_configs(
        config=config,
        custom_configs=[],
    )
    profiling_results = profile_pipeline(
        package=package,
        config=config,
        builders=builders,
        models_config=model_configs,
    )

    return profiling_results


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
        raise ModelNavigatorNotFoundError(f"Model {model_key} and runner {runner_name} not found.") from None
    runner_results.status[VerifyModel.__name__] = CommandStatus.OK


def _get_builders(framework: Framework) -> List[PipelineBuilder]:
    """Build list of pipeline builders for nav.package.optimize.

    Args:
        framework (Framework): Package framework.
    """
    conversion_builders = []
    if framework == Framework.TORCH:
        from model_navigator.pipelines.builders import torch_conversion_builder, torch_tensorrt_conversion_builder

        conversion_builders = [torch_conversion_builder, torch_tensorrt_conversion_builder]
    elif framework in (Framework.TENSORFLOW, Framework.JAX):
        from model_navigator.pipelines.builders import (
            tensorflow_conversion_builder,
            tensorflow_tensorrt_conversion_builder,
        )

        conversion_builders = [tensorflow_conversion_builder, tensorflow_tensorrt_conversion_builder]

    from model_navigator.pipelines.builders import tensorrt_conversion_builder

    conversion_builders.append(tensorrt_conversion_builder)

    builders: List[PipelineBuilder] = [
        preprocessing_builder,
        find_device_max_batch_size_builder,
        *conversion_builders,
        correctness_builder,
        performance_builder,
        verify_builder,
    ]

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
    target_formats: Optional[Tuple[Union[str, Format], ...]],
    *,
    dataloader: Optional[SizedDataLoader] = None,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    optimization_profile: Optional[OptimizationProfile] = None,
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
    if dataloader is not None:
        config.dataloader = dataloader

    # Reset profiling config
    if optimization_profile is None:
        optimization_profile = OptimizationProfile()

    config.optimization_profile = optimization_profile

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
