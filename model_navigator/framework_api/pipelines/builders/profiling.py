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

from model_navigator.framework_api.commands.load import LoadMetadata, LoadSamples
from model_navigator.framework_api.commands.performance import Performance
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import Status

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


def profiling_builder(config: Config, package_descriptor: "PackageDescriptor") -> Pipeline:

    load_metadata = LoadMetadata()
    load_samples = LoadSamples(requires=(load_metadata,))
    commands = [load_metadata, load_samples]
    for model_status in package_descriptor.navigator_status.model_status:
        for runtime_results in model_status.runtime_results:
            if runtime_results.status == Status.OK:
                commands.append(
                    Performance(
                        name=f"Performance {model_status.format.value}",
                        target_format=model_status.format,
                        requires=(load_metadata, load_samples),
                        target_jit_type=model_status.torch_jit,
                        target_precision=model_status.precision,
                        runtime_provider=runtime_results.runtime,
                    )
                )

    return Pipeline(name="Profiling", framework=package_descriptor.framework, commands=commands)
