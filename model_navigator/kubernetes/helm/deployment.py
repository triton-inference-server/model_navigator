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

from model_navigator import framework
from model_navigator.kubernetes.helm import config


class Volume(typing.NamedTuple):
    name: str
    path: str
    env: typing.Optional[str]
    empty_dir: typing.Any


class Deployment(config.Config):
    def __init__(self, name: str, entrypoint: str, framework: framework.Framework):
        self.name = name
        self.framework = framework
        self.entrypoint = entrypoint

    @property
    def volumes(self):
        return []

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
