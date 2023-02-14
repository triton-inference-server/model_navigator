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
"""Runtime search strategies."""


class RuntimeSearchStrategy:
    """Base class for runtime search strategies."""

    def __str__(self):
        """Return name of strategy."""
        return self.__class__.__name__


class MinLatencyStrategy(RuntimeSearchStrategy):
    """Get runtime with the lowest latency."""

    pass


class MaxThroughputStrategy(RuntimeSearchStrategy):
    """Get runtime with the highest throughput."""

    pass


class MaxThroughputAndMinLatencyStrategy(RuntimeSearchStrategy):
    """Get runtime with the highest throughput and the lowest latency."""

    pass


class MaxThroughputWithLatencyBudgetStrategy(RuntimeSearchStrategy):
    """Get runtime with the hightest throughput within the latency budget."""

    def __init__(self, latency_budget: float) -> None:
        """Initialize the class.

        Args:
            latency_budget: Latency budget in milliseconds.
        """
        super().__init__()
        self.latency_budget = latency_budget

    def __str__(self):
        """Return name of strategy."""
        return f"{self.__class__.__name__}({self.latency_budget}[ms])"


class SelectedRuntimeStrategy(RuntimeSearchStrategy):
    """Get a selected runtime."""

    def __init__(self, model_key: str, runner_name: str) -> None:
        """Initialize the class.

        Args:
            model_key (str): Unique key of the model.
            runner_name (str): Name of the runner.
        """
        super().__init__()
        self.model_key = model_key
        self.runner_name = runner_name

    def __str__(self):
        """Return name of strategy."""
        return f"{self.__class__.__name__}({self.model_key}:{self.runner_name})"
