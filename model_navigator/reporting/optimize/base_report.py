# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Base class for optimize report classes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pyee import EventEmitter
from rich.console import Console
from rich.table import Table

from model_navigator.reporting.optimize.events import OptimizeEvent, default_event_emitter


class BaseReport(ABC):
    """Base class for reports."""

    def __init__(self, event_emitter: Optional[EventEmitter] = None) -> None:
        """Initialized object.

        Args:
            event_emitter: optional emitter to register for events.
        """
        self.emitter = event_emitter or default_event_emitter()
        self.registered_modules = []
        self.errored_modules = []
        self.optimization_results = []
        self.current_module_name = None
        self.current_pipeline = None
        self.workspaces = []
        self.inplace_started = False
        self.has_optimization_started = False
        self.is_first_pipeline_command = False
        self.console = Console(record=True)
        self.listen_for_events()

    def listen_for_events(self):
        """Register listener on events."""
        self.emitter.on(OptimizeEvent.MODULE_REGISTERED, self.on_module_registered)
        self.emitter.on(OptimizeEvent.MODULE_REGISTRY_CLEARED, self.on_registry_cleared)
        self.emitter.on(OptimizeEvent.WORKSPACE_INITIALIZED, self.on_workspace_initialized)
        self.emitter.on(OptimizeEvent.MODULE_PICKED_FOR_OPTIMIZATION, self.on_module_picked_for_optimization)
        self.emitter.on(OptimizeEvent.OPTIMIZATION_STARTED, self.on_optimization_started)
        self.emitter.on(OptimizeEvent.OPTIMIZATION_FINISHED, self.on_optimization_finished)
        self.emitter.on(OptimizeEvent.PIPELINE_STARTED, self.on_pipeline_started)
        self.emitter.on(OptimizeEvent.PIPELINE_FINISHED, self.on_pipeline_finished)
        self.emitter.on(OptimizeEvent.COMMAND_STARTED, self.on_command_started)
        self.emitter.on(OptimizeEvent.COMMAND_FINISHED, self.on_command_finished)
        self.emitter.on(OptimizeEvent.INPLACE_STARTED, self.on_inplace_started)
        self.emitter.on(OptimizeEvent.INPLACE_FINISHED, self.on_inplace_finished)
        self.emitter.on(OptimizeEvent.BEST_MODEL_PICKED, self.on_best_model_picked)
        self.emitter.on(OptimizeEvent.MODEL_NOT_OPTIMIZED_ERROR, self.on_model_not_optimized_error)

    def on_module_registered(self, name: str, num_modules: int, num_params: int):
        """Action on module_registered event."""
        self.registered_modules.append((name, num_modules, num_params))

    def on_registry_cleared(self):
        """Action on registry cleared event."""
        self.registered_modules.clear()
        self.has_optimization_started = False
        self.workspaces = []

    def on_workspace_initialized(self, path):
        """Action on workspace initialized event."""
        self.workspaces.append(path)

    def on_inplace_started(self):
        """Action when inplace stated."""
        self.inplace_started = True
        self.optimization_results = []
        self.errored_modules = []

    def on_inplace_finished(self):
        """Action when inplace Finished."""
        self.inplace_started = False
        self.print_optimized_inplace_modules()

    def on_module_picked_for_optimization(self, name: str):
        """Action on module picked for optimization event."""
        self.current_module_name = name

    def on_pipeline_started(self, name: str):
        """Action on pipeline started event."""
        self.current_pipeline = name
        self.is_first_pipeline_command = True

    def on_pipeline_finished(self):  # noqa: B027
        """Action on pipeline finished event."""
        # nothing to be done

    @abstractmethod
    def on_optimization_started(self):
        """Action on optimization started event."""
        if not self.has_optimization_started:
            self.print_modules_table_summary()
            self.has_optimization_started = True

    def on_optimization_finished(self):
        """Action on optimization finished event."""
        # This event can be called many times in inplace. In plain mode this will be called once.
        if not self.inplace_started:
            self.print_optimized_model()

    @abstractmethod
    def on_command_started(self, command: str, config_key: Optional[str], runner_name: Optional[str]):
        """Action on command started event."""
        ...

    @abstractmethod
    def on_command_finished(self, status: str):
        """Action on command finished event."""
        ...

    def on_best_model_picked(self, config_key: str, runner_name: str, model_path: Path):
        """Action on best model picked event."""
        module_name = self.current_module_name if self.inplace_started else "model"
        self.optimization_results.append((module_name, config_key, runner_name, model_path))

    def on_model_not_optimized_error(self):
        """Action on model optimization error event."""
        if self.inplace_started:
            self.errored_modules.append(self.current_module_name)

    def print_modules_table_summary(self):
        """Prints registered modules summary."""
        if self.registered_modules:
            table = Table(title="Modules registered for optimization")
            table.add_column("Module name", style="cyan")
            table.add_column("Number of layers", justify="right", style="magenta")
            table.add_column("Number of parameters", justify="right", style="green")

            for name, num_modules, num_params in self.registered_modules:
                table.add_row(name, str(num_modules), str(num_params))

            self.console.print(table)

    def print_optimized_inplace_modules(self):
        """Prints optimization results for inplace modules."""
        self.console.print("Optimization finished for all modules.")
        if self.optimization_results:
            table = Table(title="Optimization result for max throughput and min latency strategy")
            table.add_column("Module name", style="cyan")
            table.add_column("Optimized backend", style="magenta")
            table.add_column("Path", style="green")

            for name, config_key, runner_name, model_path in self.optimization_results:
                model_path = "NA (source model is picked)" if model_path is None else str(model_path)
                table.add_row(name, f"{config_key} on {runner_name} backend", model_path)

            self.console.print(table)

        if self.errored_modules:
            table = Table(title="Could not optimize some modules")
            table.add_column("Module name", style="red")

            for name in self.errored_modules:
                table.add_row(name)

        self.save_report_to_workspace()

    def print_optimized_model(self):
        """Print optimized models for normal mode."""
        self.console.print("Optimization finished for the model.")
        if self.optimization_results:
            table = Table(title="Optimization result for max throughput and min latency strategy")
            table.add_column("Optimized backend", style="magenta")
            table.add_column("Path", style="green")

            for _, config_key, runner_name, model_path in self.optimization_results:
                model_path = "NA (source model is picked)" if model_path is None else str(model_path)
                table.add_row(f"{config_key} on {runner_name} backend", model_path)

            self.console.print(table)
        else:
            self.console.print("There were some errors during optimization and there is not any optimized model.")
        self.save_report_to_workspace()

    def save_report_to_workspace(self):
        """Saves report to the workspaces."""
        for workspace in self.workspaces:
            report_path = Path(workspace) / "optimize_report.txt"
            self.console.save_text(str(report_path), clear=False)
