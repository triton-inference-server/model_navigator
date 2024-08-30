# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

import queue
import time
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
from model_navigator.configuration.constants import (
    DEFAULT_MAX_TRIALS,
    DEFAULT_MIN_TRIALS,
    DEFAULT_STABILITY_PERCENTAGE,
    DEFAULT_STABILIZATION_WINDOWS,
    DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
)
from model_navigator.configuration.device import validate_device_string
from model_navigator.core.constants import (
    NAVIGATOR_INPLACE_OPTIMIZE_VERSION,
    NAVIGATOR_INPLACE_PROFILE_VERSION,
    NAVIGATOR_VERSION,
)
from model_navigator.core.logger import LOGGER, reconfigure_logging_to_file
from model_navigator.exceptions import ModelNavigatorModuleNotOptimizedError, ModelNavigatorRuntimeAnalyzerError
from model_navigator.inplace.config import (
    InplaceConfig as InplaceConfig,
)
from model_navigator.inplace.config import (
    OptimizeConfig,
    inplace_config,
)
from model_navigator.inplace.timer import Timer, TimerComparator  # noqa: F401
from model_navigator.inplace.wrapper import Module, module  # noqa: F401
from model_navigator.package.status import CommandStatus
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import runner_registry
from model_navigator.utils.module import lazy_import

from ..core.context import INPLACE_STRATEGIES_CONTEXT_KEY, global_context
from ..frameworks import is_torch2_available
from ..reporting.profile.events import ProfileEvent
from ..reporting.profile.events import default_event_emitter as profile_event_emitter
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
    """Optimize registered modules executed in scope of callable.

    Args:
        func:  Callable in scope of which optimize is executed.
        dataloader: List of tuples with batch size and input.
        config: Optimize config.
    """
    try:
        global_context.set(INPLACE_STRATEGIES_CONTEXT_KEY, inplace_config.strategies)
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
    except Exception as e:
        raise e
    finally:
        global_context.pop(INPLACE_STRATEGIES_CONTEXT_KEY)


def profile(
    func: Callable,
    dataloader: Sequence[Tuple[int, Any]],
    target_formats: Optional[Tuple[Union[str, Format], ...]] = None,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stability_percentage: float = DEFAULT_STABILITY_PERCENTAGE,
    stabilization_windows: int = DEFAULT_STABILIZATION_WINDOWS,
    min_trials: int = DEFAULT_MIN_TRIALS,
    max_trials: int = DEFAULT_MAX_TRIALS,
    throughput_cutoff_threshold: Optional[float] = DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    throughput_backoff_limit: int = DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    device: str = "cuda",
    initialize: bool = True,
    verbose: bool = False,
) -> InplaceProfileStatus:
    """Profile `func` in scope of which registered modules are executed.

    Args:
        func:  Callable to profile.
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
        throughput_backoff_limit: Back-off limit to run multiple more profiling steps to avoid stop at local minimum
                                  when throughput saturate based on `throughput_cutoff_threshold`.
        device: Default device used for loading unoptimized model.
        initialize: Whether to initialize pipeline on device before profiling.
        verbose: Provide verbose logging
    """
    log_file = inplace_config.cache_dir / "profiling.log"
    reconfigure_logging_to_file(log_file)

    if target_formats is None:
        target_formats = DEFAULT_TORCH_TARGET_FORMATS_FOR_PROFILING
    if runners is None:
        runners = list(runner_registry.values())

    event_emitter = profile_event_emitter()
    event_emitter.emit(ProfileEvent.PROFILING_STARTED)

    validate_device_string(device)
    modelkeys_runners = _get_modelkeys_runners(target_formats, runners)

    default_modelkeys_runners = [("python", "eager")]
    optimized_modules_count = len([m.is_optimized for m in module_registry.values()])
    if optimized_modules_count > 1:
        default_modelkeys_runners += [("navigator", "optimized")]

    modelkeys_runners = default_modelkeys_runners + list(modelkeys_runners)
    LOGGER.info(f"Profiling runners: {modelkeys_runners}")

    profiling_results = ProfilingResults()
    for model_key, runner_name in modelkeys_runners:
        runtime_name = f"{model_key} on {runner_name}"
        event_emitter.emit(ProfileEvent.RUNTIME_PROFILING_STARTED, name=runtime_name)
        try:
            _initialize_modules(
                func=func,
                model_key=model_key,
                runner_name=runner_name,
                device=device,
                initialize=initialize,
                verbose=verbose,
            )
            try:
                runner_profiling_results = RunnerProfilingResults(status=CommandStatus.OK)
                for sample_id, result in _profile_runner(
                    runner_name=runner_name,
                    func=func,
                    dataloader=dataloader,
                    min_trials=min_trials,
                    max_trials=max_trials,
                    stabilization_windows=stabilization_windows,
                    window_size=window_size,
                    stability_percentage=stability_percentage,
                    throughput_cutoff_threshold=throughput_cutoff_threshold,
                    throughput_backoff_limit=throughput_backoff_limit,
                ):
                    runner_profiling_results.detailed[sample_id] = result

                results_str = []
                for result in runner_profiling_results.detailed.values():
                    results_str.append(
                        f"""Batch: {result.batch_size:6}, """
                        f"""Throughput: {result.throughput:10.2f} [infer/sec], """
                        f"""Avg Latency: {result.avg_latency:10.2f} [ms]"""
                    )

                results_str = "\n".join(results_str)
                LOGGER.info(f"Collected results: \n{results_str}")
                time.sleep(0.1)  # FIXME: WAR to avoid overlapping messages

                for result in runner_profiling_results.detailed.values():
                    event_emitter.emit(ProfileEvent.RUNTIME_PROFILING_RESULT, result=result)

                event_emitter.emit(ProfileEvent.RUNTIME_PROFILING_FINISHED)
            except Exception as e:
                LOGGER.error(f"Profiling failed for model_key {model_key} and runner {runner_name}.")
                LOGGER.error(str(e))
                if verbose:
                    LOGGER.error(f"Traceback: {traceback.format_exc()}")

                runner_profiling_results = RunnerProfilingResults(status=CommandStatus.FAIL)
                event_emitter.emit(ProfileEvent.RUNTIME_PROFILING_ERROR)
        except Exception as e:
            LOGGER.error(f"Loading model failed for model_key {model_key} and runner {runner_name}.")
            LOGGER.error(str(e))
            if verbose:
                LOGGER.error(f"Traceback: {traceback.format_exc()}")

            runner_profiling_results = RunnerProfilingResults(status=CommandStatus.FAIL)
            event_emitter.emit(ProfileEvent.RUNTIME_PROFILING_ERROR)

        if model_key not in profiling_results.models:
            profiling_results.models[model_key] = RunnerResults()
            profiling_results.models[model_key].runners[runner_name] = RunnerProfilingResults()
        elif runner_name not in profiling_results.models[model_key].runners:
            profiling_results.models[model_key].runners[runner_name] = RunnerProfilingResults()
        profiling_results.models[model_key].runners[runner_name] = runner_profiling_results

    event_emitter.emit(ProfileEvent.PROFILING_FINISHED)

    status = _build_profile_status(profiling_results)

    return status


def _initialize_modules(func: Callable, model_key: str, runner_name: str, device: str, initialize: bool, verbose: bool):
    if initialize:
        _initialize_pipeline(func, model_key, runner_name, device)

    _load_modules(model_key, runner_name, device=device, verbose=verbose)


def _profile_runner(
    func: Callable,
    dataloader: Sequence[Tuple[int, Any]],
    runner_name: str,
    window_size: int,
    stability_percentage: float,
    stabilization_windows: int,
    min_trials: int,
    max_trials: int,
    throughput_cutoff_threshold: Optional[float],
    throughput_backoff_limit: int,
):
    if is_torch2_available():
        inference_context = torch.inference_mode
    else:
        inference_context = torch.no_grad

    prev_results = queue.Queue(maxsize=throughput_backoff_limit + 1)
    for sample_id, input_ in enumerate(dataloader):
        with NvmlHandler() as nvml_handler:
            with inference_context():
                LOGGER.debug(f"Profiling {runner_name} on sample id: {sample_id}")
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

            LOGGER.debug(
                f"Performance profiling result for {runner_name} and sample id: {sample_id}:\n{profiling_result}"
            )

            prev_result = sorted((item for _, item in prev_results.queue), key=lambda x: x.throughput, reverse=True)
            prev_result = prev_result[0] if len(prev_result) > 0 else None
            if is_throughput_saturated(profiling_result, prev_result, throughput_cutoff_threshold):
                if throughput_backoff_limit == 0:
                    break

                prev_results.put((sample_id, profiling_result))

                if prev_results.full():
                    break
            else:
                # Pop first element as is a valid result already recorded
                if not prev_results.empty():
                    prev_results.get()

                while not prev_results.empty():
                    prev_sample_id, prev_profiling_result = prev_results.get()
                    yield prev_sample_id, prev_profiling_result

                prev_results.put((sample_id, profiling_result))
                yield sample_id, profiling_result


def _build_optimize_status() -> InplaceOptimizeStatus:
    """Build optimize status."""
    modules = list(module_registry.values())
    if len(modules) == 0:
        raise ValueError("No module was found")

    m = modules[0]
    packages = m.wrapper.packages
    if len(packages) == 0:
        raise ValueError(f"Module {m.name()} has no packages")

    modules_status = {}
    for name, m in module_registry.items():
        for i, package in enumerate(m.wrapper.packages):
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
                LOGGER.info(f"Loading eager module `{module_name}` on device: `{device}`.")
                m.load_eager(device=device)
            elif model_key == "navigator" and runner_name == "optimized":
                LOGGER.info(f"Loading optimized module `{module_name}` on device: `{device}`.")
                m.load_optimized(device=device)
            else:
                LOGGER.info(
                    f"Loading optimized module `{module_name}` ({model_key}, {runner_name}) on device: `{device}`."
                )
                m.load_optimized(
                    strategies=[SelectedRuntimeStrategy(model_key=model_key, runner_name=runner_name)], device=device
                )

        except (ModelNavigatorModuleNotOptimizedError, ModelNavigatorRuntimeAnalyzerError) as e:
            LOGGER.info(f"{str(e)}" f"Loading eager module for `{module_name}` on device: `{device}`.")
            m.load_eager(device=device)
        except Exception as e:
            LOGGER.warning(f"Failed to load module {module_name} for model key {model_key} and runner {runner_name}.")
            LOGGER.warning(f"Eager module will be used on device {device}. Error message: {str(e)}")
            if verbose:
                LOGGER.warning(f"Traceback: {traceback.format_exc()}")

            LOGGER.info(f"Loading eager module `{module_name}` on device: `{device}`.")
            m.load_eager(device=device)


def _initialize_pipeline(func: Callable, model_key: str, runner_name: str, device: str) -> bool:
    if model_key == "python" and runner_name == "eager":
        return False

    if len(module_registry.values()) > 1 and hasattr(func, "to"):
        LOGGER.info("Loading eager modules.")
        for m in module_registry.values():
            m.load_eager(device=device)

        LOGGER.info(f"Initialize pipeline on device: {device}")
        func.to(device)
        return True

    return False


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

        for package in m.wrapper.packages:
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
