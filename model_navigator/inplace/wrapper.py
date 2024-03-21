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
"""Inplace Optimize model wrapper."""

import functools
from typing import Any, Callable, Optional

import wrapt

from model_navigator.inplace.timer import Timer
from model_navigator.runtime_analyzer.strategy import RuntimeSearchStrategy
from model_navigator.utils.module import lazy_import

from .config import OptimizeConfig
from .model import BaseModule, OptimizedModule, PassthroughModule, RecordAndOptimizeModule, RecordModule
from .registry import module_registry
from .utils import get_object_name

torch = lazy_import("torch")


class Module(wrapt.ObjectProxy):
    """Inplace Optimize module wrapper.

    This class wraps a torch module and provides inplace optimization functionality.
    Depending on the configuration set in config, the module will be
    optimized, recorded, or passed through.

    This wrapper can be used in place of a torch module, and will behave
    identically to the original module.

    Args:
        module: torch module to wrap.
        name: module name.
        input_mapping: function to map module inputs to the expected input.
        output_mapping: function to map module outputs to the expected output.
        offload_parameters_to_cpu: offload parameters to cpu.

    Example:
        >>> import torch
        >>> import model_navigator as nav
        >>> model = torch.nn.Linear(10, 10)
        >>> model = nav.Module(model)
    """

    def __init__(
        self,
        module: torch.nn.Module,
        name: Optional[str] = None,
        input_mapping: Optional[Callable] = None,
        output_mapping: Optional[Callable] = None,
        timer: Optional[Timer] = None,
        offload_parameters_to_cpu: bool = False,
    ) -> None:
        """Initialize Module."""
        super().__init__(module)
        self._name = name or get_object_name(module)
        self._input_mapping = input_mapping or (lambda x: x)
        self._output_mapping = output_mapping or (lambda x: x)
        self._optimize_config = OptimizeConfig()
        if timer:
            self.add_timer(timer=timer)
        else:
            self._module_timer = None

        self._wrapper = RecordAndOptimizeModule(
            module,
            # OptimizeConfig(),
            self._name,
            self._input_mapping,
            self._output_mapping,
        )
        module_registry.register(self._name, self)

    def add_timer(self, timer: Timer) -> None:
        """Add timer to module."""
        self._module_timer = timer.register_module(self._name)

    @property
    def name(self) -> str:
        """Module name."""
        return self._name

    @property
    def optimize_config(self) -> OptimizeConfig:
        """Module optimize config."""
        return self._optimize_config

    @optimize_config.setter
    def optimize_config(self, value: OptimizeConfig) -> None:
        """Module optimize config."""
        self._optimize_config = value
        self._wrapper._optimize_config = value

    def __call__(self, *args, **kwargs) -> Any:
        """Call the wrapped module.

        This method overrides the __call__ method of the wrapped module.
        If the module is already optimized it is replaced with the optimized one.
        """
        if self._module_timer and self._module_timer.enabled:
            with self._module_timer:
                output = self._wrapper(*args, **kwargs)
                if isinstance(self, torch.nn.Module):
                    torch.cuda.synchronize()
        else:
            output = self._wrapper(*args, **kwargs)

        return output

    @property
    def wrapper(self) -> BaseModule:
        """Return the wrapper module."""
        return self._wrapper

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for optimization."""
        return self.wrapper.is_ready_for_optimization

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return self.wrapper.is_optimized

    def optimize(self) -> None:
        """Optimize the module."""
        assert isinstance(self.wrapper, RecordModule), f"Module {self.name} must be in recording mode to optimize."
        assert not self.is_optimized, f"Module {self.name} is already optimized."
        assert hasattr(self.wrapper, "optimize"), f"Module {self.name} does not have an optimize method."
        self.wrapper.optimize()

    def load_optimized(self, strategy: Optional[RuntimeSearchStrategy] = None, activate_runners: bool = True) -> None:
        """Load optimized module."""
        # TODO: Consider another validation for optimization status here. is_optimized property is modified by loading passthrough.
        # if not self.is_optimized:
        #     raise ModelNavigatorModuleNotOptimizedError(f"Module {self.name} is not optimized.")
        self._wrapper = OptimizedModule(
            module=self._wrapper._module,
            # self._optimize_config,
            name=self._name,
            input_mapping=self._input_mapping,
            output_mapping=self._output_mapping,
            strategy=strategy,
            activate_runners=activate_runners,
        )

    def load_recorded(self) -> None:
        """Load recorded module."""
        self._wrapper = RecordModule(
            module=self._wrapper._module,
            name=self._name,
            input_mapping=self._input_mapping,
            output_mapping=self._output_mapping,
            optimize_config=self._optimize_config,
        )

    def load_passthrough(self) -> None:
        """Load passthrough module."""
        self._wrapper = PassthroughModule(
            module=self._wrapper._module,
            name=self._name,
            input_mapping=self._input_mapping,
            output_mapping=self._output_mapping,
            optimize_config=self._optimize_config,
        )


def module(
    module_callable: Optional[Callable[[Any], torch.nn.Module]] = None,
    name: Optional[str] = None,
    input_mapping: Optional[Callable] = None,
    output_mapping: Optional[Callable] = None,
):
    """Inplace Optimize module wrapper decorator.

    This decorator wraps a torch module and provides inplace optimization functionality.
    Depending on the configuration set in config, the module will be
    optimized, recorded, or passed through.

    This wrapper can be used in place of a torch module, and will behave
    identically to the original module.

    Args:
        module_callable: decorated callable.
        name: module name.
        input_mapping: function to map module inputs to the expected input.
        output_mapping: function to map module outputs to the expected output.

    Example:
        >>> import torch
        >>> import model_navigator as nav
        >>> @nav.module
        ... def my_model():
        ...     return torch.nn.Linear(10, 10)
        >>> model = my_model()
    """
    if module_callable is None:
        return functools.partial(
            module,
            name=name,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
        )

    @wrapt.decorator
    def wrap_module(wrapped, instance, args, kwargs):
        return Module(
            wrapped(*args, **kwargs),
            name=name,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
        )

    return wrap_module(module_callable)
