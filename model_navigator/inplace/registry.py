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
"""Inplace model registry."""

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .wrapper import Module


class ModuleRegistry:
    """Registry for inplace modules."""

    def __init__(self) -> None:
        """Initialize ModuleRegistry."""
        self._registry: Dict[str, "Module"] = {}

    def register(self, name: str, module: "Module") -> None:
        """Register a module."""
        if name in self._registry:
            raise ValueError(f"Module {name} already registered.")
        self._registry[name] = module

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
        for module in self.values():
            if not module.is_optimized:
                module.optimize()
        for module in self.values():
            module.load_optimized()

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


module_registry = ModuleRegistry()
