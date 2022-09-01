# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import shutil
from typing import TYPE_CHECKING, Callable, List, Sequence

from model_navigator.framework_api.config import Config
from model_navigator.framework_api.logger import LOGGER, add_log_file_handler
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import Indent, Status, pad_string

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


class PipelineManager:
    def __init__(self, pipeline_builders: Sequence[Callable[[Config, "PackageDescriptor"], Pipeline]]) -> None:
        self._pipeline_builders = pipeline_builders
        self._pipelines: List[Pipeline] = []

    def run(self, config: Config, package_descriptor: "PackageDescriptor") -> Sequence[Pipeline]:
        self._validate(config)
        self._prepare_workdir(config)
        self._prepare_log_file(config)

        additional_params = {}
        for pipeline_builder in self._pipeline_builders:
            pipeline = pipeline_builder(config, package_descriptor)
            additional_params = pipeline(config=config, package_descriptor=package_descriptor, **additional_params)
            self._pipelines.append(pipeline)
            package_descriptor.save_status_file()

        self._log_results()

        return self._pipelines

    @staticmethod
    def _prepare_workdir(config: Config):
        workdir = config.workdir
        if workdir.exists() and config.override_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

        workdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _prepare_log_file(config: Config):
        add_log_file_handler(log_dir=config.workdir)

    @staticmethod
    def _validate(config: Config):
        if config.workdir is None:
            raise Exception("config.workdir cannot be None")
        if config.override_workdir is None:
            raise Exception("config.override_workdir cannot be None")

    @staticmethod
    def _get_formatted_missing_paramter(param_name: str, param_desc: str):
        return f"{Indent.SINGLE}Missing parameter: {param_name}: {param_desc}"

    def _log_results(self):
        LOGGER.info(pad_string("Framework Navigator summary"))
        for pipeline in self._pipelines:
            LOGGER.info(pad_string(f"Pipeline '{pipeline.name}' summary"))
            for command in pipeline.commands:
                command_name_and_details = f"[{command.status.value:^4}] {command.get_formatted_command_details()}"
                if command.status == Status.OK:
                    LOGGER.info(command_name_and_details)
                elif command.status == Status.FAIL:
                    LOGGER.error(command_name_and_details)
                else:
                    LOGGER.warning(command_name_and_details)
                for param_name, param_desc in command.missing_params.items():
                    LOGGER.info(PipelineManager._get_formatted_missing_paramter(param_name, param_desc))
