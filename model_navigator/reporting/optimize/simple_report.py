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
"""Simple report for the optimization."""

from typing import Optional

from pyee import EventEmitter

from model_navigator.commands.base import CommandStatus
from model_navigator.pipelines.constants import (
    PIPELINE_CORRECTNESS,
    PIPELINE_FIND_MAX_BATCH_SIZE,
    PIPELINE_PERFORMANCE,
    PIPELINE_PREPROCESSING,
    PIPELINE_PROFILING,
    PIPELINE_TENSORRT_CONVERSION,
    PIPELINE_TF2_CONVERSION,
    PIPELINE_TF2_EXPORT,
    PIPELINE_TF_TENSORRT,
    PIPELINE_TORCH_CONVERSION,
    PIPELINE_TORCH_EXPORT,
    PIPELINE_TORCH_TENSORRT_CONVERSION,
    PIPELINE_VERIFY_MODELS,
)
from model_navigator.reporting.optimize.base_report import BaseReport


class SimpleReport(BaseReport):
    """Simple report."""

    def __init__(self, event_emitter: Optional[EventEmitter] = None) -> None:
        """Initialized object.

        Args:
            event_emitter: optional emitter to register for events.
        """
        super().__init__(event_emitter)
        self.status_line = None

    def on_optimization_started(self):
        """Action on optimization started event."""
        super().on_optimization_started()
        if self.current_module_name:
            self.console.print(f"[bold]Optimizing: {self.current_module_name}[/bold]")
        else:
            self.console.print("[bold]Optimization started.[/bold]")

    def on_command_started(self, command: str, config_key: Optional[str], runner_name: Optional[str]):  # noqa: C901
        """Print user friendly status from given pipeline, command."""
        if self.current_pipeline == PIPELINE_CORRECTNESS:
            self.status_line = f"Validating model {config_key} on {runner_name} backend"
        elif self.current_pipeline == PIPELINE_FIND_MAX_BATCH_SIZE:
            self.status_line = "Finding max batch size for fixed shapes based pipelines"
        elif self.current_pipeline == PIPELINE_PERFORMANCE:
            self.status_line = f"Benchmarking model {config_key} on {runner_name} backend"
        elif self.current_pipeline == PIPELINE_PREPROCESSING:
            if self.is_first_pipeline_command:
                # print just one line for all subcommands
                self.status_line = "Collecting model information"
            else:
                self.status_line = None
        elif self.current_pipeline == PIPELINE_PROFILING:
            self.status_line = "Benchmarking"
        elif self.current_pipeline == PIPELINE_TF_TENSORRT:
            self.status_line = "Building TensorFlow-TensorRT model from TensorFlow SavedModel"
        elif self.current_pipeline == PIPELINE_TF2_CONVERSION:
            if command == "ConvertSavedModel2ONNX":
                self.status_line = "Building ONNX model from TensorFlow SavedModel"
            elif command == "GraphSurgeonOptimize":
                self.status_line = f"Optimizing graph for {config_key}"
        elif self.current_pipeline == PIPELINE_TF2_EXPORT:
            self.status_line = "Building TensorFlow SavedModel from TensorFlow model"
        elif self.current_pipeline == PIPELINE_TENSORRT_CONVERSION:
            self.status_line = f"Building TensorRT engine {config_key} from ONNX model"
        elif self.current_pipeline == PIPELINE_TORCH_TENSORRT_CONVERSION:
            self.status_line = "Building Torch-TensorRT model from ExportedProgram model"
        elif self.current_pipeline == PIPELINE_TORCH_EXPORT:
            if command == "GraphSurgeonOptimize":
                self.status_line = f"Optimizing graph for {config_key}"
            elif command == "ExportTorch2TorchScript":
                self.status_line = f"Building {config_key} model from Torch model"
            elif command == "ExportExportedProgram":
                self.status_line = "Building ExportedProgram from Torch model"
            elif command == "ExportTorch2ONNX" and config_key == "onnx-dynamo":
                self.status_line = "Building ONNX Dynamo model from Torch model"
            elif command == "ExportTorch2ONNX" and config_key == "onnx":
                self.status_line = "Building ONNX Trace model from Torch model"
        elif self.current_pipeline == PIPELINE_TORCH_CONVERSION:
            if command == "GraphSurgeonOptimize":
                self.status_line = f"Optimizing graph for {config_key}"
            elif command == "ConvertTorchScript2ONNX":
                self.status_line = "Building ONNX model from TorchScript model"
        elif self.current_pipeline == PIPELINE_VERIFY_MODELS:
            self.status_line = f"Verifying model {config_key} on {runner_name} backend"

        if self.status_line:
            module_name = self.current_module_name if self.current_module_name else "Model"
            self.status_line = f"[bold]{module_name}:[/bold] " + self.status_line
            self.console.print(f"{self.status_line} ...")

    def on_command_finished(self, status: str):
        """Action on command finished event."""
        # update last status
        if self.status_line is not None:
            status = CommandStatus(status)
            if status == CommandStatus.OK:
                style = "green"
            elif status == CommandStatus.FAIL:
                style = "red"
            else:
                style = "black"
            self.console.print(f"{self.status_line} [bold {style}]{status.value}[/bold {style}]")

        self.is_first_pipeline_command = False
