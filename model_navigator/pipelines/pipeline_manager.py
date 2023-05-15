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
import shutil
import uuid
from typing import Dict, List, Optional, Sequence

from tabulate import tabulate

from model_navigator.api.config import Format, TensorRTProfile
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.core.constants import NAVIGATOR_PACKAGE_VERSION, NAVIGATOR_VERSION
from model_navigator.core.package import Package
from model_navigator.core.status import ModelStatus, Status
from model_navigator.core.tensor import TensorMetadata
from model_navigator.logger import LOGGER, add_log_file_handler
from model_navigator.pipelines.builders import PipelineBuilder
from model_navigator.pipelines.validation import PipelineManagerConfigurationValidator
from model_navigator.utils.common import pad_string
from model_navigator.utils.environment import get_env


class PipelineManager:
    """Class for managing pipelines runs."""

    _common_config_filter_fields = [
        "model",
        "model_params",
        "dataloader",
        "workspace",
        "forward_kw_names",
        "verify_func",
    ]

    @classmethod
    def run(
        cls,
        pipeline_builders: Sequence[PipelineBuilder],
        config: CommonConfig,
        package: Optional[Package] = None,
        models_config: Optional[Dict[Format, List[ModelConfig]]] = None,
    ) -> Package:
        """Run pipeline manager and build a package.

        Args:
            pipeline_builders: List of pipelines builders to run.
            config: A configuration object
            package: Package to update, if None new package is build. Defaults to None.
            models_config: List of model configs to use in pipelines builders. Defaults to None.

        Returns:
            Package object
        """
        if config.verbose or config.debug:
            config.log()

        PipelineManagerConfigurationValidator.run(config, package)

        package = cls._prepare_package(package, config)

        cls._prepare_log_file(config)
        cls._run_pipelines_builders(
            pipeline_builders=pipeline_builders, models_config=models_config, config=config, package=package
        )
        package.status.config = config.to_dict(
            cls._common_config_filter_fields,
            parse=True,
        )

        package.save_status_file()

        LOGGER.warning(
            "Initially models are not verified. Validate exported models and use "
            "PackageDescriptor.set_verified(format, runtime, jit_type, precision) method to set models as verified."
        )

        package._forward_kw_names = config.forward_kw_names

        return package

    @classmethod
    def _new_package(cls, config: CommonConfig):
        status = Status(
            uuid=str(uuid.uuid1()),
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            environment=get_env(),
            config={},
            models_status={},
            input_metadata=TensorMetadata(),
            output_metadata=TensorMetadata(),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=1,
        )

        package = Package(status, config.workspace, model=config.model)

        return package

    @classmethod
    def _from_package(cls, package: Package):
        status = Status(
            uuid=str(uuid.uuid1()),
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            environment=get_env(),
            config={},
            models_status={},
            input_metadata=TensorMetadata(),
            output_metadata=TensorMetadata(),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=1,
        )

        package = Package(status, workspace=package.workspace, model=package.model)

        return package

    @classmethod
    def _prepare_package(cls, package: Optional[Package], config: CommonConfig) -> Package:
        if package is None:
            package = cls._new_package(config)
            cls._prepare_workspace(config)
        else:
            package = cls._from_package(package)

        return package

    @classmethod
    def _run_pipelines_builders(
        cls,
        pipeline_builders: Sequence[PipelineBuilder],
        config: CommonConfig,
        package: Package,
        models_config: Optional[Dict[Format, List[ModelConfig]]] = None,
    ) -> None:
        shared_parameters = {}
        for pipeline_builder in pipeline_builders:
            pipeline = pipeline_builder(config, models_config)
            shared_parameters = pipeline.run(config=config, package=package, **shared_parameters)
            package.save_status_file()

        cls._log_model_status(package.status.models_status)

    @staticmethod
    def _log_model_status(models_status: Dict[str, ModelStatus]):
        summary = [[] for _ in range(len(models_status))]
        model_status, runner_status = None, None
        for i, model_status in enumerate(models_status.values()):
            summary[i].append(model_status.model_config.format.value)
            summary[i].append(model_status.model_config.key)
            if model_status.model_config.parent_key:
                summary[i].append(model_status.model_config.parent_key)
            else:
                summary[i].append("framework")
            summary[i].append("\n".join([f"{k}: {v.value}" for k, v in model_status.status.items()]))
            runtime_status = []

            for runner_status in model_status.runners_status.values():
                runtime_status.append(
                    "\n".join(
                        [runner_status.runner_name] + [f"    {k}: {v.value}" for k, v in runner_status.status.items()]
                    )
                )
            summary[i].append("\n".join(runtime_status))

        headers = ["Format", "Key", "Parent model key"]
        if model_status:
            headers.append("Model status")
        if runner_status:
            headers.append("Runner status")
        table = tabulate(summary, headers, "grid")
        LOGGER.info(f"\n{pad_string('Model Navigator Summary')}\n{table}")

    @staticmethod
    def _prepare_workspace(config: CommonConfig):
        workspace = config.workspace
        if workspace.exists():
            LOGGER.info(f"Removing exiting workspace at {workspace}")
            shutil.rmtree(workspace, ignore_errors=True)

        workspace.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _prepare_log_file(config: CommonConfig):
        add_log_file_handler(log_dir=config.workspace)
