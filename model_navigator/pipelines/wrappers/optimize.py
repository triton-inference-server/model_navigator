# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Execute optimize pipelines and return package."""

import pathlib
from typing import Any, Dict, List, Optional, Sequence

from pyee import EventEmitter

from model_navigator.configuration import Format
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model import model_config
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorRuntimeAnalyzerError, ModelNavigatorRuntimeError
from model_navigator.package.builder import PackageBuilder
from model_navigator.package.package import Package
from model_navigator.pipelines.builders import PipelineBuilder
from model_navigator.pipelines.pipeline_manager import PipelineManager
from model_navigator.reporting.events import NavigatorEvent, default_event_emitter


def optimize_pipeline(
    workspace: pathlib.Path,
    builders: Sequence[PipelineBuilder],
    config: CommonConfig,
    model: Optional[Any] = None,
    package: Optional[Package] = None,
    models_config: Optional[Dict[Format, List[model_config.ModelConfig]]] = None,
) -> Package:
    """Execute optimize pipeline which returns package.

    Args:
        workspace: Workspace where execution is performed
        builders: Pipeline builders
        config: Common configuration
        model: Model to optimize
        package: Package to optimize
        models_config: Model configs used during optimize

    Returns:
        Package with data and results snapshot
    """
    if not model and not package:
        raise ModelNavigatorRuntimeError("`model` or `package` is required.")

    if model and package:
        raise ModelNavigatorRuntimeError("Only one of `model` and `package` arguments is required.")

    event_emitter = default_event_emitter()

    workspace = Workspace(workspace)
    if not package or workspace.path != package.workspace.path:
        workspace.initialize()
        event_emitter.emit(NavigatorEvent.WORKSPACE_INITIALIZED, path=workspace.path)

    if package:
        model = package.model

    pipeline_manager = PipelineManager(workspace=workspace)
    event_emitter.emit(NavigatorEvent.OPTIMIZATION_STARTED)
    context = pipeline_manager.run(
        workspace=workspace,
        builders=builders,
        config=config,
        models_config=models_config,
        package=package,
    )

    package_builder = PackageBuilder()
    package = package_builder.create(
        model=model,
        context=context,
        config=config,
    )

    _emit_optimization_result_event(package, event_emitter)

    event_emitter.emit(NavigatorEvent.OPTIMIZATION_FINISHED)
    return package


def _emit_optimization_result_event(package: Package, event_emitter: EventEmitter):
    """Emits event with the best result or error."""
    try:
        best_model_status = package.get_best_model_status(include_source=True)
        best_format_path = package.workspace.path / best_model_status.model_config.path
        model_path = best_format_path if best_format_path.exists() else None
        event_emitter.emit(
            NavigatorEvent.BEST_MODEL_PICKED,
            config_key=best_model_status.model_config.key,
            runner_name=list(best_model_status.runners_status.keys())[0],
            model_path=model_path,
        )
    except ModelNavigatorRuntimeAnalyzerError:
        event_emitter.emit(NavigatorEvent.MODEL_NOT_OPTIMIZED_ERROR)
