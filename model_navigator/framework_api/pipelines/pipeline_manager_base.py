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
from typing import List

from model_navigator.framework_api.config import Config
from model_navigator.framework_api.logger import LOGGER, add_log_file_handler
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import Indent, Status, get_package_path, pad_string


class PipelineManager:
    def _get_pipeline(self, config) -> Pipeline:
        raise NotImplementedError

    def build(self, config) -> PackageDescriptor:
        self._validate(config)
        self._prepare_package_dir(config)
        self._prepare_log_file(config)

        pipelines = [self._get_pipeline(config)]

        for pipeline in pipelines:
            pipeline(config=config)

        self._log_results(pipelines)

        # pytype: disable=bad-return-type
        return PackageDescriptor.from_pipelines(pipelines, config)
        # pytype: enable=bad-return-type

    @staticmethod
    def _prepare_package_dir(config: Config):
        package_dir_path = get_package_path(config.workdir, config.model_name)
        if package_dir_path.exists() and config.override_workdir:
            shutil.rmtree(package_dir_path, ignore_errors=True)
        package_dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _prepare_log_file(config: Config):
        add_log_file_handler(log_dir=get_package_path(config.workdir, config.model_name))

    @staticmethod
    def _validate(config: Config):
        if config.workdir is None:
            raise Exception("config.workdir cannot be None")
        if config.override_workdir is None:
            raise Exception("config.override_workdir cannot be None")

    @staticmethod
    def _get_formatted_missing_paramter(param_name: str, param_desc: str):
        return f"{Indent.SINGLE}Missing parameter: {param_name}: {param_desc}"

    def _log_results(self, pipelines: List[Pipeline]):
        LOGGER.info(pad_string("Framework Navigator summary"))
        for pipeline in pipelines:
            LOGGER.info(pad_string(f"Pipeline {pipeline.name} summary"))
            for command in pipeline.commands:
                command_name_and_details = f"[{command.status.value:^4}] {command.get_formatted_command_details()}"
                if command.status == Status.OK:
                    LOGGER.info(command_name_and_details)
                elif command.status == Status.FAIL:
                    LOGGER.error(command_name_and_details)
                else:
                    LOGGER.warning(command_name_and_details)
                for param_name, param_desc in command.missing_params.items():
                    LOGGER.info(self._get_formatted_missing_paramter(param_name, param_desc))
