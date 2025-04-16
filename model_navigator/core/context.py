# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
"""Execution context shared between operations."""

from collections import OrderedDict
from typing import Any

INPLACE_OPTIMIZE_KEY = "inplace_optimize"
INPLACE_OPTIMIZE_STRATEGIES_CONTEXT_KEY = "inplace_optimize_strategies"
INPLACE_OPTIMIZE_BATCH_CONTEXT_KEY = "inplace_optimize_batch"
INPLACE_OPTIMIZE_MODULE_NAME_CONTEXT_KEY = "inplace_optimize_current_module_name"
INPLACE_OPTIMIZE_WORKSPACE_CONTEXT_KEY = "inplace_optimize_workspace"
INPLACE_OPTIMIZE_MODULE_GRAPH_ID_CONTEXT_KEY = "inplace_optimize_module_graph_id"


class GlobalContext:
    """Context for sharing global state."""

    def __init__(self) -> None:
        """Initialize global context."""
        self._context: OrderedDict[str, Any] = OrderedDict()

    def set(self, name: str, value: Any):
        """Set value in context."""
        self._context[name] = value

    def get(self, name: str) -> Any:
        """Get a value from context."""
        return self._context.get(name)

    def pop(self, name: str) -> Any:
        """Pop a value from context."""
        return self._context.pop(name)

    def clear(self) -> None:
        """Removes already values from context.

        Warning: this should only be called when you want to optimize already registered modules once again
        from scratch.
        """
        self._context = OrderedDict()

    def temporary(self) -> "TemporaryGlobalContext":
        """Create a temporary context."""
        return TemporaryGlobalContext(self)


class TemporaryGlobalContext:
    """Keeps track of variable changes and restores values of global context after exit of the scope."""

    def __init__(self, global_context: GlobalContext):
        """Inits."""
        self._gc = global_context
        self._changes = {}

    def __enter__(self) -> "TemporaryGlobalContext":
        """Enter context scope."""
        return self

    def set(self, name: str, value: Any):
        """Setting context values and remembering previous ones."""
        if name not in self._changes:  # keep only first occurrence
            self._changes[name] = self._gc.get(name)
        self._gc.set(name, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores values or removes them if None."""
        for name, value in self._changes.items():
            if value is None:
                self._gc.pop(name)
            else:
                self._gc.set(name, value)


global_context = GlobalContext()
