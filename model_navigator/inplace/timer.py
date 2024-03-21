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
# noqa: D104
"""Inplace Optimize utilities."""

import contextlib
import dataclasses
import pathlib
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import dacite
import tabulate
import yaml

from model_navigator.api.config import Format
from model_navigator.core.constants import DEFAULT_COMPARISON_REPORT_FILE
from model_navigator.inplace.registry import module_registry
from model_navigator.utils.environment import get_env

from .config import inplace_config

format2string = {
    Format.TORCH: "Torch",
    Format.TORCHSCRIPT: "TorchScript",
    Format.TF_SAVEDMODEL: "SavedModel",
    Format.ONNX: "ONNX",
    Format.TENSORRT: "TensorRT plan",
    Format.TORCH_TRT: "TorchTensorRT",
    Format.TF_TRT: "SavedModel",
    Format.JAX: "JAX",
    Format.PYTHON: "Python",
}


@dataclasses.dataclass(order=True)
class RuntimeResults:
    """Dataclass used for storing module time data."""

    average_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    call_count: Optional[int] = None


@dataclasses.dataclass(order=True)
class ModuleTimeData:
    """Dataclass used for storing module time data."""

    module_name: str
    formats: List[str]
    runners: List[str]
    runtime_results: Optional[RuntimeResults] = None
    times: Optional[List[float]] = dataclasses.field(default_factory=list)

    @property
    def total_time_ms(self) -> float:
        """Get total time spent in the __call__ method."""
        return self.runtime_results.total_time_ms

    @property
    def average_time_ms(self) -> float:
        """Get average time spent in the __call__ method."""
        return self.runtime_results.average_time_ms

    @property
    def call_count(self) -> int:
        """Get number of calls to the pipeline."""
        return self.runtime_results.call_count

    def __post_init__(self):
        """Post init."""
        self.runtime_results = RuntimeResults(
            total_time_ms=sum(self.times),
            average_time_ms=sum(self.times) / len(self.times) if self.times else 0,
            call_count=len(self.times),
        )

    @classmethod
    def from_dict(cls, data_dict: Dict):
        """Create from dict."""
        return cls(**data_dict)


@dataclasses.dataclass(order=True)
class TimeData:
    """Dataclass used for storing pipeline time data."""

    name: str
    info: Optional[Dict[str, str]] = None
    runtime_results: Optional[RuntimeResults] = None
    times: Optional[List[float]] = dataclasses.field(default_factory=list)
    modules: Optional[Dict[str, ModuleTimeData]] = None

    def __post_init__(self):
        """Post init."""
        self.runtime_results = RuntimeResults(
            total_time_ms=sum(self.times),
            average_time_ms=sum(self.times) / len(self.times) if self.times else 0,
            call_count=len(self.times),
        )

    @property
    def total_time_ms(self) -> float:
        """Get total time spent in the pipeline."""
        return self.runtime_results.total_time_ms

    @property
    def average_time_ms(self) -> float:
        """Get average time spent in the pipeline."""
        return self.runtime_results.average_time_ms

    @property
    def call_count(self) -> int:
        """Get number of calls to the pipeline."""
        return self.runtime_results.call_count

    @property
    def modules_total_times(self) -> Dict[str, float]:
        """Get time spent in the __call__ method of each module."""
        return {name: module_stats.total_time_ms for name, module_stats in self.modules.items()}

    @property
    def modules_average_times(self) -> Dict[str, float]:
        """Get average time spent in the __call__ method of each module."""
        return {name: module_stats.average_time_ms for name, module_stats in self.modules.items()}

    @property
    def modules_measurements_count(self) -> Dict[str, int]:
        """Get number of measurements for each module."""
        return {name: len(module_stats.times) for name, module_stats in self.modules.items()}

    @property
    def measurements_count(self) -> int:
        """Get number of measurements."""
        return len(self.times)

    @property
    def module_timers(self) -> Dict[str, ModuleTimeData]:
        """Get module timers."""
        return self.modules

    @property
    def module_names(self):
        """Get module names."""
        return list(self.modules.keys())

    @classmethod
    def get_save_path(cls) -> pathlib.Path:
        """Get save path."""
        return pathlib.Path(inplace_config.cache_dir) / "time_data.yaml"

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data_dict: Dict):
        """Create from dict."""
        return dacite.from_dict(cls, data_dict)

    def save(self):
        """Save to json."""
        with open(self.get_save_path(), "w") as fp:
            yaml.dump(self.to_dict(), fp, sort_keys=False)

    @classmethod
    def load(cls):
        """Load from json."""
        with open(cls.get_save_path()) as fp:
            data = yaml.safe_load(fp)
        return dacite.from_dict(cls, data)


class ModuleTimer(contextlib.AbstractContextManager):
    """Timer context manager for measuring time spent in the __call__ method of a module."""

    def __init__(self, module_name: str) -> None:
        """Initialize ModuleTimer."""
        super().__init__()
        self._enabled = False
        self._module_name = module_name
        self._times = []

    def __enter__(self) -> Any:
        """Enter context."""
        self._start = time.monotonic()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: F841
        """Exit context."""
        if self._enabled:
            self._times.append((time.monotonic() - self._start) * 1000)  # convert to ms

    def enable(self):
        """Enable module timers."""
        self._enabled = True

    def disable(self):
        """Disable module timers."""
        self._enabled = False

    @property
    def enabled(self):
        """Check if module timers are enabled."""
        return self._enabled

    @property
    def module_name(self) -> str:
        """Module name."""
        return self._module_name

    @property
    def module_formats(self):
        """Get module formats."""
        module_wrapper = module_registry.get(name=self._module_name).wrapper
        if hasattr(module_wrapper, "_runners") and module_wrapper._runners:
            nav_formats = [runner.format() for runner in module_wrapper._runners.values()]
        else:
            # TODO: Currently inline supports only PyTorch so in passthrough mode only torch format is supported.
            nav_formats = [Format.TORCH]

        return [format2string[nav_format] for nav_format in nav_formats]

    @property
    def module_runners(self):
        """Get module runners."""
        module_wrapper = module_registry.get(name=self._module_name).wrapper
        if hasattr(module_wrapper, "_runners") and module_wrapper._runners:
            return [runner.name() for runner in module_wrapper._runners.values()]
        else:
            return []

    @property
    def time_data(self) -> ModuleTimeData:
        """Get module timer data."""
        return ModuleTimeData(
            module_name=self._module_name,
            times=self._times,
            formats=self.module_formats,
            runners=self.module_runners,
        )

    def reset(self):
        """Reset the total time spent in the __call__ method."""
        self._times = []


class Timer(contextlib.AbstractContextManager):
    """Timer context manager.

    This class is used to measure the time spent in the __call__ method of the wrapped modules.
    In order to use it, it needs to be passed to the Module constructor.

    Additionally it can be used as a context manager to measure the time spent in the context
    and measure whole pipeline execution time.

    Args:
        name: Timer name.
    """

    def __init__(self, name: str, info: Optional[Dict[str, str]] = None) -> None:
        """Initialize Timer."""
        super().__init__()
        self._module_timers = {}
        self._times = []
        self._name = name
        self._info = info

    def __enter__(self) -> Any:
        """Enter context."""
        for module_timer in self._module_timers.values():
            module_timer.enable()
        self._start = time.monotonic()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: F841
        """Exit context."""
        self._times.append((time.monotonic() - self._start) * 1000)  # convert to ms
        for module_timer in self._module_timers.values():
            module_timer.disable()
        self.save()

    @property
    def name(self) -> str:
        """Timer name."""
        return self._name

    @property
    def _time_data(self) -> TimeData:
        """Get timer data."""
        return TimeData(
            name=self._name,
            modules={name: module_timer.time_data for name, module_timer in self._module_timers.items()},
            times=self._times,
            info=self._info,
        )

    def save(self, comparison_report_path: Optional[pathlib.Path] = None):
        """Save to json timer data and comparison report.

        Comparison report is generated only if the pipeline was executed in both modes: run and passthrough.

        Args:
            comparison_report_path: Path where the comparison report will be saved. Default: report.yaml
        """
        if not comparison_report_path:
            comparison_report_path = pathlib.Path(DEFAULT_COMPARISON_REPORT_FILE)

        self._time_data.save()

        if TimeData.get_save_path().exists() and TimeData.get_save_path().exists():
            timer_comparator = TimerComparator()
            with open(comparison_report_path, "w") as fp:
                fp.write(timer_comparator.get_report())

    def register_module(self, module_name: str) -> ModuleTimer:
        """Register a module in timer."""
        if module_name in self._module_timers:
            raise ValueError(f"Module {module_name} already registered.")
        self._module_timers[module_name] = ModuleTimer(module_name)

        return self._module_timers[module_name]

    def reset(self):
        """Reset the total time spent in the context."""
        self._times = []
        for module_timer in self._module_timers.values():
            module_timer.reset()


class TimerComparator:
    """Utility for comparing two timers."""

    def __init__(self) -> None:
        """Initialize TimerComparator."""
        super().__init__()
        # TODO: Fix missing mode parameter
        self._original_timer_data = TimeData.load()
        self._optimized_timer_data = TimeData.load()

    @property
    def total_speedup(self) -> float:
        """Get total speedup of the optimized pipeline."""
        if not self._optimized_timer_data.total_time_ms:
            return 0
        return self._original_timer_data.total_time_ms / self._optimized_timer_data.total_time_ms

    @property
    def framework_coverage(self):
        """Get framework coverage."""
        module_contrib_to_total_time = {}
        frameworks_coverage = Counter()
        for module_name, total_time in self._original_timer_data.modules_total_times.items():
            module_contrib_to_total_time[module_name] = total_time / self._original_timer_data.total_time_ms

        for module_name, contribution in module_contrib_to_total_time.items():
            module_formats = self._optimized_timer_data.module_timers[module_name].formats
            for module_format in module_formats:
                divider = len(module_formats)
                frameworks_coverage[module_format] += contribution / divider

        return frameworks_coverage

    @property
    def original_total_time(self):
        """Get original total time."""
        return self._original_timer_data.total_time_ms

    @property
    def optimized_total_time(self):
        """Get optimized total time."""
        return self._optimized_timer_data.total_time_ms

    def _remove_key_from_nested_dict(self, d: Dict, key: str):
        """Remove key from nested dict."""
        d.pop(key, None)
        for v in d.values():
            if isinstance(v, dict):
                self._remove_key_from_nested_dict(v, key)

    def format_dict_recursive(self, data, level=0):
        """Format dict recursively."""
        table = []

        for key, value in data.items():
            if isinstance(value, dict):
                table.append([f"{'  ' * level}{key}:", ""])
                table.extend(self.format_dict_recursive(value, level + 1))
            elif isinstance(value, list):
                table.append([f"{'  ' * level}{key}:", ", ".join(value)])
            elif isinstance(value, float):
                table.append([f"{'  ' * level}{key}:", f"{value:.2f}"])
            else:
                table.append([f"{'  ' * level}{key}:", value])

        return table

    def get_report(self):
        """Get report."""
        tabulate.PRESERVE_WHITESPACE = True
        d = self._optimized_timer_data.to_dict()
        d["runtime_results"]["navigator_speedup"] = f"{self.total_speedup:.2f}x"
        d["runtime_results"]["navigator_runtime_coverage"] = {
            framework: f"{float(coverage * 100):.2f}%" for framework, coverage in self.framework_coverage.items()
        }

        not_covered = 1 - sum(self.framework_coverage.values())
        d["runtime_results"]["not_covered"] = f"{float(not_covered * 100):.2f}%"
        self._remove_key_from_nested_dict(d, "times")
        d["environment"] = get_env()

        t = self.format_dict_recursive(d)

        output = tabulate.tabulate(t, tablefmt="plain")
        return output

    def save_report(self, path: Union[str, pathlib.Path]):
        """Save report."""
        with open(path, "w") as fp:
            fp.write(self.get_report())
