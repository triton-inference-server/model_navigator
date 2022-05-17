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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from model_navigator.framework_api.commands.core import Command
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import Framework, pad_string

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


class Pipeline:
    def __init__(
        self,
        name: str,
        commands: List[Command],
        framework: Optional[Framework] = None,
    ):
        self.name = name
        self.id = name.lower().replace(" ", "_").replace("-", "_")
        self.framework = framework
        self.commands = commands

    def __call__(self, config: Config, package_descriptor: "PackageDescriptor", **kwargs) -> Dict[str, Any]:
        LOGGER.info(pad_string(f"Pipeline '{self.name}' started"))
        additional_params = kwargs
        for command in self.commands:
            LOGGER.info(pad_string(command.get_formatted_command_details()))
            command.transform(package_descriptor, **{**config.to_dict(), **additional_params})

            output_names = command.get_output_name()
            outputs = command.output
            if output_names is not None:
                if outputs is None:
                    outputs = ()
                if isinstance(output_names, str):
                    output_names, outputs = (output_names,), (outputs,)
                for output_name, output in zip(output_names, outputs):
                    additional_params[output_name] = output
                    if output_name in config.__dataclass_fields__:
                        setattr(config, output_name, output)
        return additional_params
