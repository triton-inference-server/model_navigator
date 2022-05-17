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


from typing import TYPE_CHECKING

from model_navigator.framework_api.commands.config_cli import ConfigCli
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import Status

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


def config_generation_builder(config: Config, package_descriptor: "PackageDescriptor") -> Pipeline:
    commands = []
    for model_status in package_descriptor.navigator_status.model_status:
        if any(runtime_results.status == Status.OK for runtime_results in model_status.runtime_results):
            commands.append(
                ConfigCli(
                    name=f"Generate configurations for {model_status.format.value}",
                    target_format=model_status.format,
                    target_jit_type=model_status.torch_jit,
                    target_precision=model_status.precision,
                )
            )

    return Pipeline(
        name="Generate Configurations",
        framework=package_descriptor.framework,
        commands=commands,
    )
