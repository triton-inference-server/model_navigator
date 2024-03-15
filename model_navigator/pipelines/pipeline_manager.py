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
"""Pipeline manager submodule."""

from typing import Dict, List, Optional, Sequence

from model_navigator.api.config import Format
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.core.logger import LOGGER, log_dict
from model_navigator.core.workspace import Workspace
from model_navigator.package.package import Package
from model_navigator.pipelines.builders import PipelineBuilder
from model_navigator.pipelines.pipeline_context import PipelineContext
from model_navigator.pipelines.validation import PipelineManagerConfigurationValidator


class PipelineManager:
    """Class for managing pipelines runs."""

    _common_config_filter_fields = [
        "model",
        "model_params",
        "dataloader",
        "workspace",
        "verify_func",
    ]

    def __init__(self, workspace: Workspace):
        """Initialize PipelineManager."""
        self._workspace = workspace

    def run(
        self,
        workspace: Workspace,
        builders: Sequence[PipelineBuilder],
        config: CommonConfig,
        package: Optional[Package] = None,
        models_config: Optional[Dict[Format, List[ModelConfig]]] = None,
    ) -> PipelineContext:
        """Run pipeline manager and build a package.

        Args:
            workspace: Workspace where unit is executed
            builders: List of pipelines builders to run.
            config: A configuration object
            package: Package to update, if None new package is build. Defaults to None.
            models_config: List of model configs to use in pipelines builders. Defaults to None.

        Returns:
            Package object
        """
        if config.verbose or config.debug:
            data = config.to_dict(
                filter_fields=[
                    "model",
                    "dataloader",
                    "verify_func",
                ],
                parse=True,
            )
            log_dict(title="Common config parameters", data=data)

        PipelineManagerConfigurationValidator.run(config, package)

        context = PipelineContext(workspace=self._workspace)
        context.initialize()

        pipelines = self._build_pipelines(
            builders=builders,
            models_config=models_config,
            config=config,
        )

        for pipeline in pipelines:
            pipeline.run(workspace=workspace, config=config, context=context)

        context.log_status()

        LOGGER.warning(
            "Initially models are not verified. Validate exported models and use "
            "PackageDescriptor.set_verified(format, runtime, jit_type, precision) method to set models as verified."
        )

        return context

    def _build_pipelines(
        self,
        builders: Sequence[PipelineBuilder],
        config: CommonConfig,
        models_config: Optional[Dict[Format, List[ModelConfig]]] = None,
    ) -> List:
        pipelines = []
        for pipeline_builder in builders:
            pipeline = pipeline_builder(config, models_config)
            pipelines.append(pipeline)

        return pipelines
