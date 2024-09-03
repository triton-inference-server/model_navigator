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
"""Inplace model registry."""

import gc
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict

import model_navigator.core.context as ctx
from model_navigator.reporting.optimize.events import OptimizeEvent, default_event_emitter

if TYPE_CHECKING:
    from .wrapper import Module


class ModuleRegistry:
    """Registry for inplace modules."""

    def __init__(self) -> None:
        """Initialize ModuleRegistry."""
        self._registry: OrderedDict[str, Module] = OrderedDict()
        self.event_emitter = default_event_emitter()

    def register(self, name: str, module: "Module") -> None:
        """Register a module."""
        if name in self._registry:
            raise ValueError(f"Module {name} already registered.")
        self._registry[name] = module
        self._emit_module_registered(name, module)

    def clear(self) -> None:
        """Removes already registered modules.

        Warning: this should only be called when you want to optimize already registered modules once again
        from scratch.
        """
        self._registry = OrderedDict()
        gc.collect()
        self.event_emitter.emit(OptimizeEvent.MODULE_REGISTRY_CLEARED)

    @property
    def modules(self) -> Dict[str, "Module"]:
        """Get all registered modules."""
        return self._registry

    def get(self, name: str) -> "Module":
        """Get a module."""
        return self._registry[name]

    def check_all_ready(self) -> bool:
        """Check if all registered modules have enough samples."""
        for module in self._registry.values():
            if not module.is_optimized and not module.is_ready_for_optimization:
                return False
        return True

    def optimize(self) -> None:
        """Optimize all registered modules."""
        self.event_emitter.emit(OptimizeEvent.INPLACE_STARTED)
        for name, module in self.items():
            if not module.is_optimized:
                ctx.global_context.set(ctx.INPLACE_OPTIMIZE_MODULE_NAME_CONTEXT_KEY, name)
                self.event_emitter.emit(OptimizeEvent.MODULE_PICKED_FOR_OPTIMIZATION, name=name)
                module.optimize()
        self.event_emitter.emit(OptimizeEvent.INPLACE_FINISHED)

    def items(self):
        """Return registered items."""
        return self._registry.items()

    def keys(self):
        """Return registered keys."""
        return self._registry.keys()

    def values(self):
        """Return registered values."""
        return self._registry.values()

    def is_empty(self):
        """Return True if registry is empty."""
        return not bool(self._registry)

    def _emit_module_registered(self, name, module: "Module"):
        """Emits event about module being registered."""
        num_modules = len(list(module.modules()))
        num_params = sum(p.numel() for p in module.parameters())

        self.event_emitter.emit(
            OptimizeEvent.MODULE_REGISTERED,
            name=name,
            num_modules=num_modules,
            num_params=num_params,
        )


module_registry = ModuleRegistry()
