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
from dataclasses import dataclass
from typing import List, Optional

from model_navigator.utils.docker import (
    ContainerIdType,
    ContainerIpAddress,
    ContainerMount,
    DockerContainer,
    get_docker_container_id,
)


@dataclass(frozen=True)
class EnvironmentState:
    docker_container_id: Optional[ContainerIdType] = None
    docker_container_ip_address: Optional[ContainerIpAddress] = None
    docker_container_mounts: Optional[List[ContainerMount]] = None


def get_environment_state() -> EnvironmentState:
    mounts = None
    docker_container_id = get_docker_container_id()

    if docker_container_id:
        from docker.errors import DockerException

        try:
            container = DockerContainer(docker_container_id)
            mounts = container.mounts
        except DockerException:
            pass

    return EnvironmentState(
        docker_container_id=docker_container_id,
        docker_container_mounts=mounts,
    )
