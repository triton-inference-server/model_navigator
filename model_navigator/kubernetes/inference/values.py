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
from typing import NamedTuple

from collections import Iterable

from .. import helm
from ..triton import TritonServer
from ..utils import format_value


class Values(helm.Values):
    def __init__(self, container_version: str, parameters: NamedTuple):
        self.container_version = container_version
        self.parameters = parameters

    def data(self):
        values = dict(
            imagePullSecret=None,
            pullPolicy="Always",
            restartPolicy="Always",
            replicaCount=1,
            gpu=dict(
                limit=1,
                product=None,
                mig=None,
            ),
            deployer=dict(
                image=None,
                modelUri=None,
                gcsCredentialsFile=None,
                awsCredentialsFile=None,
                azureCredentialsFile=None,
            ),
        )

        values["server"] = {"image": TritonServer.container_image(version=self.container_version)}

        for key, value in zip(self.parameters._fields, self.parameters):
            key = format_value(key)

            if isinstance(value, Iterable) and not isinstance(value, str):
                value = " ".join(map(str, value))

            values["deployer"][key] = value

        values["service"] = {
            "type": "ClusterIP",
        }

        return values
