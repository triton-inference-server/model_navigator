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
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import dockerpty
from docker import APIClient, from_env
from docker.errors import BuildError
from docker.types import DeviceRequest

LOGGER = logging.getLogger(__name__)


class DockerContainer:
    def __init__(self, container_id: str):
        self._docker_api_client = APIClient()
        self._docker_client = from_env()
        containers = self._docker_client.containers.list(filters={"id": container_id})
        if not containers:
            raise ValueError(f"Could not find container with id={container_id}")
        if len(containers) > 1:
            raise ValueError(f"More than 1 docker container matches id={container_id}")

        self._container = containers[0]

    def run_cmd(
        self,
        cmd: str,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ):
        LOGGER.debug(f"Running cmd: {cmd}")
        dockerpty.exec_command(
            self._docker_api_client, self._container.id, command=cmd, stdin=stdin, stdout=stdout, stderr=stderr
        )

    @property
    def id(self):
        return self._container.id

    def kill(self):
        self._container.kill()

    def get_port_binding_host_ip(self, port):
        self._container.reload()
        host_bindings = self._container.ports[f"{port}/tcp"]
        LOCALHOST_IP = "127.0.0.1"
        host_ip = [binding["HostIp"] for binding in host_bindings][0] or LOCALHOST_IP
        if host_ip == "0.0.0.0":
            host_ip = LOCALHOST_IP
        return host_ip

    @property
    def ip_address(self):
        self._container.reload()
        return self._container.attrs["NetworkSettings"]["IPAddress"]


class DockerImage:
    def __init__(self, image_name: str):
        self._image_name = image_name
        self._docker_client = from_env()
        self._docker_api_client = APIClient()

    def run_container(
        self,
        *,
        workdir_path: Optional[Path] = None,
        devices: Optional[List[DeviceRequest]] = None,
        environment: Optional[Dict[str, str]] = None,
        mount_as_volumes: Optional[List[Path]] = None,
    ):
        devices = devices or []
        mount_as_volumes = mount_as_volumes or []
        environment = environment or {}
        volumes = {p.resolve().as_posix(): {"bind": p.resolve().as_posix(), "mode": "rw"} for p in mount_as_volumes}
        return self._run_container(devices=devices, volumes=volumes, environment=environment, workdir_path=workdir_path)

    def _fix_mounted_dirs_ownerships(self, container, mount_as_volumes):
        import os

        current_uid = os.geteuid()
        current_gid = os.getegid()
        for mounted_dir in mount_as_volumes:
            LOGGER.debug(f"Fixing ownership of {mounted_dir}")
            cmd = f"chown -R {current_uid}:{current_gid} {mounted_dir}"
            dockerpty.exec_command(self._docker_api_client, container.id, command=cmd)

    def _run_container(
        self,
        *,
        workdir_path: Optional[Path] = None,
        devices: Optional[List[DeviceRequest]] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        environment: Optional[Dict[str, str]] = None,
    ):
        LOGGER.info(f"Run docker container with image {self._image_name}; using workdir: {workdir_path}")
        devices = devices or []
        for device in devices:
            LOGGER.debug(f"Using device: {device}")
        volumes = volumes or {}
        for volume_src in volumes:
            LOGGER.debug(f"Mounting volume: {volume_src}")
        environment = environment or {}
        LOGGER.debug(f"Setting environment: {environment}")
        container = self._docker_client.containers.run(
            image=self._image_name,
            device_requests=devices,
            environment=environment,
            volumes=volumes,
            stdin_open=True,
            tty=True,
            stream=True,
            detach=True,
            auto_remove=True,
            working_dir=workdir_path.as_posix() if workdir_path else None,
        )
        LOGGER.debug(f"Started docker container {container.id[:8]}")

        return DockerContainer(container.id)

    def exists(self) -> bool:
        images = list(self._docker_client.images.list(self._image_name))
        exists = bool(images)
        LOGGER.debug(f"Image {self._image_name} {'exists' if exists else 'is missing'}")
        return exists


class DockerBuilder:
    def __init__(self):
        self._docker_client = from_env()

    def build(
        self,
        *,
        dockerfile_path: Path,
        image_name: str,
        workdir_path: Optional[Path] = None,
        build_args: Optional[Dict[str, Any]] = None,
    ) -> DockerImage:
        workdir_path = workdir_path or dockerfile_path.parent
        build_args = build_args or {}
        LOGGER.info(f"Building {image_name} docker image.")
        LOGGER.debug(f"    Using workdir: {workdir_path}")
        LOGGER.debug(f"    Dockerfile: {dockerfile_path}")
        LOGGER.debug(f"    Build args: {build_args}")
        build_logs = list()
        try:
            _, build_logs = self._docker_client.images.build(
                path=workdir_path.resolve().as_posix(),
                dockerfile=dockerfile_path.resolve().as_posix(),
                tag=image_name,
                buildargs=build_args,
                network_mode="host",
                rm=True,
            )
        except BuildError as e:
            build_logs = e.build_log
            raise e
        finally:
            for chunk in build_logs:
                log = chunk.get("stream")
                if log:
                    LOGGER.debug(log.rstrip())

        return DockerImage(image_name)
