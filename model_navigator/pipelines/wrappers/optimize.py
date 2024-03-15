# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

from model_navigator.api.config import Format
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model import model_config
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorRuntimeError
from model_navigator.package.builder import PackageBuilder
from model_navigator.package.package import Package
from model_navigator.pipelines.builders import PipelineBuilder
from model_navigator.pipelines.pipeline_manager import PipelineManager


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

    workspace = Workspace(workspace)
    if not package or workspace.path != package.workspace.path:
        workspace.initialize()

    if package:
        model = package.model

    pipeline_manager = PipelineManager(workspace=workspace)
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

    return package
