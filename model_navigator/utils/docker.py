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
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

LOGGER = logging.getLogger(__name__)

ContainerIdType = str
ContainerIpAddress = str
CONTAINER_ID_LENGTH = 10


@dataclass(frozen=True, order=True)
class ContainerMount:
    host_path: Path
    container_path: Path
    mount_type: str


class DockerContainer:
    def __init__(self, container_id: ContainerIdType):
        from docker import APIClient, from_env

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
        detached: bool = False,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ):
        LOGGER.debug(f"Running cmd: {cmd}")
        interactive = stdin.isatty() if hasattr(stdin, "isatty") else False
        import dockerpty

        for text_io in [stdout, stderr]:
            # dockerpty requires either fileno attribute or send method from stdout/stderr
            if text_io and isinstance(text_io, io.StringIO):
                text_io.send = lambda b: text_io.write(b.decode("UTF-8"))
        if detached:
            _, logs_gen = self._container.exec_run(cmd=cmd, stream=True)
            return logs_gen
        else:
            dockerpty.exec_command(
                self._docker_api_client,
                self._container.id,
                command=cmd,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                interactive=interactive,
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

    @property
    def mounts(self) -> List[ContainerMount]:
        self._container.reload()
        mounts = self._container.attrs["Mounts"]
        return [
            ContainerMount(
                host_path=Path(mount["Source"]),
                container_path=Path(mount["Destination"]),
                mount_type=mount["Type"],
            )
            for mount in mounts
        ]


class DockerImage:
    def __init__(self, image_name: str):
        from docker import APIClient, from_env

        self._image_name = image_name
        self._docker_client = from_env()
        self._docker_api_client = APIClient()

    def run_container(
        self,
        *,
        workdir_path: Optional[Path] = None,
        devices: Optional[List] = None,  # docker.types.DeviceRequest
        environment: Optional[Dict[str, str]] = None,
        mount_as_volumes: Optional[List[Path]] = None,
        ports: Optional[Union[Dict[int, int], List[int]]] = None,
    ):
        devices = devices or []
        environment = environment or {}
        volumes = self._get_volumes(mount_as_volumes)
        ports = ports or []
        return self._run_container(
            devices=devices,
            volumes=volumes,
            environment=environment,
            ports=ports,
            workdir_path=workdir_path,
        )

    def _get_volumes(self, mount_as_volumes: Optional[List[Path]]):

        mount_as_volumes = self._get_volume_mappings_containing(mount_as_volumes)
        mount_as_volumes = sorted(set(mount_as_volumes or []))
        volumes = {
            mount.host_path.resolve().as_posix(): {"bind": mount.container_path.resolve().as_posix()}
            for mount in mount_as_volumes
        }
        return volumes

    def _get_volume_mappings_containing(self, container_side_required_paths: List[Path]) -> List[ContainerMount]:
        from model_navigator.utils.env import get_environment_state

        def _is_relative_to(a: Union[str, Path], *outer):
            try:
                a = Path(a)
                a.relative_to(*outer)
                return True
            except ValueError:
                return False

        environment_state = get_environment_state()
        is_model_navigator_running_in_container = environment_state.docker_container_id is not None

        result = []
        for container_side_required_path in container_side_required_paths:
            if is_model_navigator_running_in_container:
                mounts = [
                    mount
                    for mount in environment_state.docker_container_mounts
                    if _is_relative_to(container_side_required_path, mount.container_path)
                ]
                assert len(mounts) == 1, f"for {container_side_required_path} found {mounts}"
                mount = mounts[0]
            else:
                mount = ContainerMount(
                    host_path=container_side_required_path,
                    container_path=container_side_required_path,
                    mount_type="bind",
                )
            result.append(mount)
        mounts = sorted(set(result))
        return mounts

    def _run_container(
        self,
        *,
        workdir_path: Optional[Path] = None,
        devices: Optional[List] = None,  # docker.types.DeviceRequest
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        ports: Optional[List[Union[int, Tuple[int, int]]]] = None,
        environment: Optional[Dict[str, str]] = None,
    ):
        LOGGER.info(f"Run docker container with image {self._image_name}; using workdir: {workdir_path}")
        devices = devices or []
        for device in devices:
            LOGGER.debug(f"Using device: {device}")
        volumes = volumes or {}
        for host_path, volume_spec in volumes.items():
            LOGGER.debug(f"Mounting volume: {host_path}:{volume_spec['bind']}")
        environment = environment or {}
        ports = ports or []
        if not isinstance(ports, dict):
            ports = {port: None for port in ports}  # on host side bind to random port

        LOGGER.debug(f"Setting environment: {environment}")
        container = self._docker_client.containers.run(
            image=self._image_name,
            device_requests=devices,
            environment=environment,
            volumes=volumes,
            ports=ports,
            stdin_open=True,
            tty=True,
            stream=True,
            detach=True,
            auto_remove=True,
            ipc_mode="host",
            working_dir=workdir_path.as_posix() if workdir_path else None,
            user=os.getuid(),
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
        from docker import from_env

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
        build_logs = []

        from docker.errors import BuildError

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


def get_docker_container_id() -> Optional[ContainerIdType]:
    """Return container id in which current process is running or None if process is not running in container"""
    try:
        cpuset_path = Path("/proc/1/cpuset")
        cpuset_content = cpuset_path.read_text("utf-8")
        path = Path(cpuset_content).name[:CONTAINER_ID_LENGTH].strip()
        return path or None
    except FileNotFoundError:
        return None
