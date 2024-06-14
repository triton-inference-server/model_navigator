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
"""Inplace Optimize API."""

import traceback
import uuid
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import model_navigator.inplace.bundle as bundle  # noqa: F401
from model_navigator.commands.correctness.correctness import Correctness
from model_navigator.commands.performance.nvml_handler import NvmlHandler
from model_navigator.commands.performance.performance import Performance
from model_navigator.commands.performance.utils import is_throughput_saturated
from model_navigator.configuration import (
    DEFAULT_TORCH_TARGET_FORMATS_FOR_PROFILING,
    Format,
    SelectedRuntimeStrategy,
    TensorRTPrecision,
)
from model_navigator.configuration.validation.device import validate_device_string
from model_navigator.core.constants import (
    DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD,
    NAVIGATOR_INPLACE_OPTIMIZE_VERSION,
    NAVIGATOR_INPLACE_PROFILE_VERSION,
    NAVIGATOR_VERSION,
)
from model_navigator.core.logger import LOGGER, pad_string
from model_navigator.exceptions import ModelNavigatorModuleNotOptimizedError, ModelNavigatorRuntimeAnalyzerError
from model_navigator.inplace.config import InplaceConfig, OptimizeConfig, inplace_config  # noqa: F401
from model_navigator.inplace.timer import Timer, TimerComparator  # noqa: F401
from model_navigator.inplace.wrapper import Module, module  # noqa: F401
from model_navigator.package.status import CommandStatus
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import runner_registry
from model_navigator.utils.module import lazy_import

from ..utils.environment import get_env
from .profiling import ProfilingResults, RunnerProfilingResults, RunnerResults, run_measurement
from .registry import module_registry
from .status import InplaceOptimizeStatus, InplaceProfileStatus, ModuleStatus

torch = lazy_import("torch")


def load_optimized(device: Union[str, "torch.device"] = "cuda"):
    """Load optimized modules.

    Args:
        device: Device on which optimized models are loaded.
    """
    for m in module_registry.values():
        m.load_optimized(device=device)


def optimize(
    func: Callable,
    dataloader: Sequence[Tuple[int, Any]],
    config: Optional[OptimizeConfig] = None,
) -> InplaceOptimizeStatus:
    """Optimize all registered modules.

    Args:
        func: Function to optimize.
        dataloader: List of tuples with batch size and input.
        config: Optimize config.
    """
    if config is None:
        config = OptimizeConfig()

    for m in module_registry.values():
        # set main config if user did not provide one for a module
        if m.optimize_config is None:
            m.optimize_config = config
        m.load_recorded()

    for input_ in dataloader:
        _, sample = input_  # unpack batch_size and sample
        if not isinstance(sample, (list, tuple)):
            sample = (sample,)
        if not isinstance(sample[-1], dict):
            sample = (*sample, {})
        *args, kwargs = sample
        func(*args, **kwargs)

    module_registry.optimize()
    return _build_optimize_status()


def profile(
    func: Callable,
    dataloader: Sequence[Tuple[int, Any]],
    target_formats: Optional[Tuple[Union[str, Format], ...]] = None,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    min_trials: int = 3,
    max_trials: int = 10,
    stabilization_windows: int = 3,
    window_size: int = 50,
    stability_percentage: float = 10.0,
    throughput_cutoff_threshold: Optional[float] = DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD,
    device: str = "cuda",
    verbose: bool = False,
) -> InplaceProfileStatus:
    """Profile `func` and all registered modules.

    Args:
        func: Function to profile.
        dataloader: List of tuples with batch size and input.
        target_formats: Target model formats for optimize process
        runners: Use only runners provided as parameter
        min_trials: Minimum number of trials.
        max_trials: Maximum number of trials.
        stabilization_windows: Number of stabilization windows.
        window_size: Number of inference queries performed in measurement window
        stability_percentage: Allowed percentage of variation from the mean in three consecutive windows.
        throughput_cutoff_threshold: Minimum throughput increase to continue profiling. If None is provided,
                                     profiling run through whole dataloader
        device: Default device used for loading unoptimized model.
        verbose: Provide verbose logging
    """
    if target_formats is None:
        target_formats = DEFAULT_TORCH_TARGET_FORMATS_FOR_PROFILING
    if runners is None:
        runners = list(runner_registry.values())

    validate_device_string(device)

    modelkeys_runners = _get_modelkeys_runners(target_formats, runners)

    LOGGER.info(f"Profiling runners: {modelkeys_runners}")

    modelkeys_runners = [("python", "eager")] + list(modelkeys_runners)

    profiling_results = ProfilingResults()
    for model_key, runner_name in modelkeys_runners:
        LOGGER.info(pad_string(f"Profiling of {model_key} and {runner_name}"))

        _load_modules(model_key, runner_name, device=device, verbose=verbose)
        runner_profiling_results = RunnerProfilingResults()
        try:
            prev_result = None
            for sample_id, input_ in enumerate(dataloader):
                with NvmlHandler() as nvml_handler:
                    LOGGER.info(f"Profiling {runner_name} on sample id: {sample_id}")
                    profiling_result = run_measurement(
                        func=func,
                        sample=input_,
                        nvml_handler=nvml_handler,
                        min_trials=min_trials,
                        max_trials=max_trials,
                        stabilization_windows=stabilization_windows,
                        window_size=window_size,
                        stability_percentage=stability_percentage,
                    )

                    LOGGER.info(
                        f"Performance profiling result for {runner_name} "
                        f"and sample id: {sample_id}:\n{profiling_result}"
                    )
                    runner_profiling_results.detailed[sample_id] = profiling_result

                    if is_throughput_saturated(profiling_result, prev_result, throughput_cutoff_threshold):
                        break
                    prev_result = profiling_result

            runner_profiling_results.status = CommandStatus.OK
        except Exception as e:
            LOGGER.error(f"Profiling failed for model_key {model_key} and runner {runner_name}.")
            LOGGER.error(str(e))
            if verbose:
                LOGGER.error(f"Traceback: {traceback.format_exc()}")

            runner_profiling_results.status = CommandStatus.FAIL

        if model_key not in profiling_results.models:
            profiling_results.models[model_key] = RunnerResults()
            profiling_results.models[model_key].runners[runner_name] = RunnerProfilingResults()
        elif runner_name not in profiling_results.models[model_key].runners:
            profiling_results.models[model_key].runners[runner_name] = RunnerProfilingResults()
        profiling_results.models[model_key].runners[runner_name] = runner_profiling_results

    status = _build_profile_status(profiling_results)

    return status


def _build_optimize_status() -> InplaceOptimizeStatus:
    """Build optimize status."""
    modules = list(module_registry.values())
    if len(modules) == 0:
        raise ValueError("No module was found")

    m = modules[0]
    packages = m._wrapper._packages
    if len(packages) == 0:
        raise ValueError(f"Module {m.name()} has no packages")

    modules_status = {}
    for name, m in module_registry.items():
        for i, package in enumerate(getattr(m._wrapper, "_packages", [])):
            module_name = f"{name}.{i}"
            module_status = ModuleStatus.from_package_status(package.status)
            modules_status[module_name] = module_status

    status = InplaceOptimizeStatus(
        status_version=NAVIGATOR_INPLACE_OPTIMIZE_VERSION,
        model_navigator_version=NAVIGATOR_VERSION,
        uuid=str(uuid.uuid1()),
        environment=get_env(),
        module_status=modules_status,
    )
    return status


def _build_profile_status(profiling_results: ProfilingResults) -> InplaceProfileStatus:
    """Build profile status."""
    status = InplaceProfileStatus(
        status_version=NAVIGATOR_INPLACE_PROFILE_VERSION,
        model_navigator_version=NAVIGATOR_VERSION,
        uuid=str(uuid.uuid1()),
        environment=get_env(),
        profiling_results=profiling_results,
    )
    return status


def _load_modules(model_key: str, runner_name: str, device: str, verbose: bool = False):
    for module_name, m in module_registry.items():
        try:
            if model_key == "python" and runner_name == "eager":
                m.load_eager()
            else:
                m.load_optimized(
                    strategy=SelectedRuntimeStrategy(model_key=model_key, runner_name=runner_name), device=device
                )

        except (ModelNavigatorModuleNotOptimizedError, ModelNavigatorRuntimeAnalyzerError) as e:
            LOGGER.info(f"{str(e)}" f"Loading eager module.")
            m.load_eager(device=device)
        except Exception as e:
            LOGGER.warn(f"Failed to load module {module_name} for model key {model_key} and runner {runner_name}.")
            LOGGER.warn(f"Eager module will be used. Error message: {str(e)}")
            if verbose:
                LOGGER.warn(f"Traceback: {traceback.format_exc()}")

            m.load_eager(device=device)


def _format_to_modelkey(format: Union[str, Format]):
    if isinstance(format, Format):
        format = format.value
    if format in (Format.TENSORRT, Format.TENSORRT.value):
        return tuple(f"trt-{p.value}" for p in TensorRTPrecision)
    return (format,)


def _get_modelkeys_runners(formats, runners):
    if runners and isinstance(runners[0], Type):
        runners = [runner.name() for runner in runners]
    if formats and isinstance(formats[0], Format):
        formats = [format.value for format in formats]

    modelkeys = set()
    for format in formats:
        modelkeys.update(_format_to_modelkey(format))

    modelkeys_runners = set()
    for name, m in module_registry.items():
        try:
            m.load_optimized(activate_runners=False)
        except ModelNavigatorModuleNotOptimizedError as e:
            raise ModelNavigatorModuleNotOptimizedError(
                f"Module {name} not optimized. Please optimize the nav.optimize command first."
            ) from e
        for package in m._wrapper._packages:
            for modelkey, model_status in package.status.models_status.items():
                for runner_name, runner_status in model_status.runners_status.items():
                    if (
                        runner_status.status.get(Correctness.__name__)
                        == runner_status.status.get(Performance.__name__)
                        == CommandStatus.OK
                        and modelkey in modelkeys
                        and runner_name in runners
                    ):
                        modelkeys_runners.add((modelkey, runner_name))

    modelkeys_runners = sorted(modelkeys_runners)
    return modelkeys_runners
