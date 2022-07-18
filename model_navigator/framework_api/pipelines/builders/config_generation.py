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


from typing import TYPE_CHECKING, Dict, List, Tuple

from model_navigator.framework_api.commands.config_cli import ConfigCli
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.status import ModelStatus
from model_navigator.framework_api.utils import Status

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


def _calculate_max_tolerance(config: Config, model_statuses: List[ModelStatus]) -> Tuple[Dict, Dict]:
    """
    Collect the tolerance configuration for CLI comparator. Select the biggest value for not failed runtimes or
    values provided in export configuration.
    """
    atol = {}
    rtol = {}

    def select_tolerance(name, d, value):
        if name not in d:
            return value

        return value if value > d[name] else d[name]

    for model_status in model_statuses:
        for runtime_result in model_status.runtime_results:
            if runtime_result.status != Status.OK:
                continue

            if not runtime_result.tolerance:
                continue

            for name, tol in runtime_result.tolerance.items():
                atol[name] = config.atol if config.atol else select_tolerance(name, atol, tol.atol)
                rtol[name] = config.atol if config.atol else select_tolerance(name, rtol, tol.rtol)

    return atol, rtol


def config_generation_builder(config: Config, package_descriptor: "PackageDescriptor") -> Pipeline:
    commands = []
    atol, rtol = _calculate_max_tolerance(config, package_descriptor.navigator_status.model_status)
    for model_status in package_descriptor.navigator_status.model_status:
        if any(runtime_results.status == Status.OK for runtime_results in model_status.runtime_results):
            commands.append(
                ConfigCli(
                    name=f"Generate configurations for {model_status.format.value}",
                    target_format=model_status.format,
                    target_jit_type=model_status.torch_jit,
                    target_precision=model_status.precision,
                    runtime_results=model_status.runtime_results,
                    atol=atol,
                    rtol=rtol,
                )
            )

    return Pipeline(
        name="Generate Configurations",
        framework=package_descriptor.framework,
        commands=commands,
    )
