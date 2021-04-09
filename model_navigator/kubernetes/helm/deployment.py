# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import typing

from ... import framework
from .. import utils
from . import config


class Volume(typing.NamedTuple):
    name: str
    path: str
    env: typing.Optional[str]
    empty_dir: typing.Any


class Deployment(config.Config):
    def __init__(self, name: str, entrypoint: str, framework: framework.Framework, parameters: typing.NamedTuple):
        self.name = name
        self.framework = framework
        self.parameters = parameters
        self.entrypoint = entrypoint

        if not hasattr(self.parameters, "model_name"):
            raise ValueError("Parameters need to have model_name.")

    @property
    def volumes(self):
        return []

    def _prepare_environment(self, env: typing.List, namespace: typing.Optional[str] = None):
        for key in self.parameters._fields:
            env_var = utils.format_env(key)
            param = utils.format_value(key)

            value_str = ".Values"
            if namespace:
                value_str += f".{namespace}"

            env.append({"name": env_var, "value": f"{{{{ quote {value_str}.{param} }}}}"})

    def _prepare_volumes(self, env: typing.List, volumeAttach: typing.List, volumeMounts: typing.List):
        for volume in self.volumes:
            if volume.env is not None:
                env.append(
                    {
                        "name": volume.env,
                        "value": volume.path,
                    },
                )

            volumeMounts.append({"name": volume.name, "mountPath": volume.path})

            volumeAttach.append(
                {
                    "name": volume.name,
                    "emptyDir": volume.empty_dir,
                },
            )
