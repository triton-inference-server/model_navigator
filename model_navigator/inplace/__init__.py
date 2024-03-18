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

import copy
import pathlib
import traceback
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import yaml

from model_navigator.api.config import DEFAULT_TORCH_TARGET_FORMATS_FOR_PROFILING, Format
from model_navigator.commands.correctness.correctness import Correctness
from model_navigator.commands.performance.nvml_handler import NvmlHandler
from model_navigator.commands.performance.performance import Performance
from model_navigator.core.logger import LOGGER, pad_string
from model_navigator.exceptions import ModelNavigatorModuleNotOptimizedError
from model_navigator.package.status import CommandStatus
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import runner_registry
from model_navigator.runtime_analyzer.strategy import SelectedRuntimeStrategy
from model_navigator.utils.common import DataObject

from .config import OptimizeConfig
from .profiling import ProfilingResults, RunnerProfilingResults, RunnerResults, run_measurement
from .registry import module_registry

OPTIMIZATION_STATUS_FILE_NAME = "optimization_status.yaml"


def load_optimized():
    """Load optimized modules."""
    for module in module_registry.values():
        module.load_optimized()


def optimize(
    func: Callable,
    dataloader: Sequence[Tuple[int, Any]],
    config: Optional[OptimizeConfig] = None,
    status_path: Optional[Union[pathlib.Path, str]] = None,
) -> None:
    """Optimize all registered modules.

    Args:
        func: Function to optimize.
        dataloader: List of tuples with batch size and input.
        config: Optimize config.
        status_path: Path to store the optimization status.
    """
    if config is None:
        config = OptimizeConfig()

    for module in module_registry.values():
        module.optimize_config = config
        module.load_recorded()

    for input_ in dataloader:
        _, sample = input_  # unpack batch_size and sample
        if not isinstance(sample, tuple):
            sample = (sample,)
        if not isinstance(sample[-1], dict):
            sample = (*sample, {})
        *args, kwargs = sample
        func(*args, **kwargs)

    module_registry.optimize()
    _build_optimize_status(status_path)


PROFILING_RESULTS_FILE_NAME = "profiling_status.yaml"


def profile(
    func: Callable,
    dataloader: Sequence[Tuple[int, Any]],
    target_formats: Optional[Tuple[Union[str, Format], ...]] = None,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    status_path: Optional[Union[pathlib.Path, str]] = None,
    min_trials: int = 3,
    max_trials: int = 10,
    stabilization_windows: int = 3,
    window_size: int = 50,
    stability_percentage: float = 10.0,
    verbose: bool = False,
) -> None:
    """Profile `func` and all registered modules.

    Args:
        func: Function to profile.
        dataloader: List of tuples with batch size and input.
        target_formats: Target model formats for optimize process
        runners: Use only runners provided as parameter
        status_path: Path to store the profiling results.
        min_trials: Minimum number of trials.
        max_trials: Maximum number of trials.
        stabilization_windows: Number of stabilization windows.
        window_size: Number of inference queries performed in measurement window
        stability_percentage: Allowed percentage of variation from the mean in three consecutive windows.
        verbose: Provide verbose logging
    """
    if target_formats is None:
        target_formats = DEFAULT_TORCH_TARGET_FORMATS_FOR_PROFILING
    if runners is None:
        runners = list(runner_registry.values())

    modelkeys_runners = _get_modelkeys_runners(target_formats, runners)

    LOGGER.info(f"Profiling runners: {modelkeys_runners}")

    modelkeys_runners = [("python", "eager")] + list(modelkeys_runners)

    profiling_results = ProfilingResults()
    for model_key, runner_name in modelkeys_runners:
        LOGGER.info(pad_string(f"Profiling of {model_key} and {runner_name}"))

        _load_modules(model_key, runner_name, verbose=verbose)
        runner_profiling_results = RunnerProfilingResults()
        try:
            for sample_id, input_ in enumerate(dataloader):
                with NvmlHandler() as nvml_handler:
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

    if status_path is None:
        status_path = pathlib.Path.cwd() / PROFILING_RESULTS_FILE_NAME
    elif isinstance(status_path, str):
        status_path = pathlib.Path(status_path)

    profiling_results.to_file(status_path)


def _build_optimize_status(optimization_status_path):
    """Build optimize status."""
    module = list(module_registry.values())[0]
    pkg = getattr(module._wrapper, "_packages", [])[0]
    status = _status_serializable_dict(pkg.status)
    optimize_status = {
        "format_version": status["format_version"],
        "model_navigator_version": status["model_navigator_version"],
        "environment": status["environment"],
        "config": status["config"],
        "module_status": {},
    }

    for name, module in module_registry.items():
        for i, package in enumerate(getattr(module._wrapper, "_packages", [])):
            module_name = f"{name}.{i}"
            module_status = _status_serializable_dict(package.status)
            module_status.pop("format_version")
            module_status.pop("model_navigator_version")
            module_status.pop("environment")
            module_status.pop("config")
            optimize_status["module_status"][module_name] = module_status

    if optimization_status_path is None:
        optimization_status_path = pathlib.Path.cwd() / OPTIMIZATION_STATUS_FILE_NAME
    elif isinstance(optimization_status_path, str):
        optimization_status_path = pathlib.Path(optimization_status_path)

    with open(optimization_status_path, "w") as f:
        yaml.safe_dump(optimize_status, f, sort_keys=False)


def _status_serializable_dict(status) -> Dict:
    """Convert status to serializable dict."""
    config = DataObject.filter_data(
        data=status.config,
        filter_fields=[
            "model",
            "dataloader",
            "verify_func",
            "workspace",
        ],
    )
    config = DataObject.parse_data(config)
    status = copy.copy(status)
    status.config = config
    data = status.to_dict(parse=True)
    return data


def _load_modules(model_key: str, runner_name: str, verbose: bool = False):
    for module_name, module in module_registry.items():
        try:
            if model_key == "python" and runner_name == "eager":
                module.load_passthrough()
                module._wrapper._module.to("cuda")  # TODO: remove this line when passthrough is fixed
            else:
                module.load_optimized(strategy=SelectedRuntimeStrategy(model_key=model_key, runner_name=runner_name))
                if (
                    model_key == "torch" and "cpu" not in runner_name.lower()
                ):  # TODO: remove after fixing torch runner device handling
                    module._wrapper._module.to("cuda")

        except ModelNavigatorModuleNotOptimizedError:
            LOGGER.info(
                f"Module {module_name} not optimized for modelkey {model_key} and runner {runner_name}. "
                f"Unoptimized module will be used."
            )
            module.load_passthrough()
            module._wrapper._module.to("cuda")  # TODO: remove this line when passthrough is fixed
        except Exception as e:
            LOGGER.warn(f"Failed to load module {module_name} for model key {model_key} and runner {runner_name}.")
            LOGGER.warn(f"Unoptimized module will be used. Error message: {str(e)}")
            if verbose:
                LOGGER.warn(f"Traceback: {traceback.format_exc()}")

            module.load_passthrough()
            module._wrapper._module.to("cuda")  # TODO: remove this line when passthrough is fixed


def _format_to_modelkey(format: Union[str, Format]):
    if isinstance(format, Format):
        format = format.value
    if format in (Format.TENSORRT, Format.TENSORRT.value):
        return ("trt-fp16", "trt-fp32")
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
    for name, module in module_registry.items():
        try:
            module.load_optimized(activate_runners=False)
        except ModelNavigatorModuleNotOptimizedError as e:
            raise ModelNavigatorModuleNotOptimizedError(
                f"Module {name} not optimized. Please optimize the nav.optimize command first."
            ) from e
        for package in module._wrapper._packages:
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
    return modelkeys_runners
