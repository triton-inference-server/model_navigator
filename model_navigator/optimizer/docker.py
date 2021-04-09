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
from typing import IO, Any, Dict, Generator, List, Optional, Set, Union

import logging
from pathlib import Path
from threading import Thread

import docker
from docker.errors import BuildError, DockerException
from docker.models.containers import Container
from docker.types import DeviceRequest

_MODEL_NAVIGATOR_DIR = Path(__file__).parent.parent.parent
LOGGER = logging.getLogger("model_navigator.optimizer")


class Docker:
    def __init__(self, dockerfile_path: Path):
        self._dockerfile_path = dockerfile_path
        self._docker_client = docker.from_env()
        self._docker_api_client = docker.APIClient()

    def build(self, build_args: Dict[str, Any]):
        image_name = build_args["FROM_IMAGE_NAME"]
        new_image_name = self._get_new_docker_image_name(image_name)

        stream = list()
        try:
            _, stream = self._docker_client.images.build(
                path=_MODEL_NAVIGATOR_DIR.resolve().as_posix(),
                dockerfile=self._dockerfile_path.resolve().as_posix(),
                tag=new_image_name,
                buildargs=build_args,
                network_mode="host",
            )
        except BuildError as e:
            stream = e.build_log
            raise e
        finally:
            for chunk in stream:
                log = chunk.get("stream")
                if log:
                    LOGGER.debug(log.rstrip())

        return new_image_name

    def _get_new_docker_image_name(self, image_name):
        *_, base_name, image_tag = image_name.split(":")
        base_name = base_name.split("/")[-1]
        new_image_name = f"model_navigator_optimizer_{base_name}:{image_tag}"
        return new_image_name

    def run(
        self,
        cmd: str,
        *,
        image_name: str,
        devices: List[DeviceRequest],
        log_writer,
        workdir: Path,
        env: Optional[Dict[str, str]] = None,
        mount_as_volumes: Optional[Set] = None,
        verbose: bool = False,
    ):
        mount_as_volumes = mount_as_volumes or []
        env = env or {}
        volumes = {p.resolve().as_posix(): {"bind": p.resolve().as_posix(), "mode": "rw"} for p in mount_as_volumes}

        LOGGER.info("Building optimizer docker image")
        new_image_name = self.build(build_args={"FROM_IMAGE_NAME": image_name})

        LOGGER.info("Run optimizer")
        container = self._run_container(new_image_name, devices, volumes, env)
        try:
            self._exec_in_container(container=container, cmd=cmd, workdir=workdir, log_writer=log_writer)
        except DockerException as e:
            raise e
        finally:
            container.kill()

    def _run_container(
        self, image_name: str, devices: List[DeviceRequest], volumes: Dict, environment: Optional[Dict[str, str]]
    ):
        container = self._docker_client.containers.run(
            image=image_name,
            device_requests=devices,
            environment=environment,
            volumes=volumes,
            tty=True,
            stream=True,
            detach=True,
        )

        return container

    def _exec_in_container(self, container: Container, cmd: Union[str, List], workdir: Path, log_writer) -> Dict:
        exec = self._docker_api_client.exec_create(
            container=container.id, cmd=cmd, workdir=workdir.resolve().as_posix()
        )
        logger = self._docker_api_client.exec_start(exec_id=exec["Id"], stream=True)
        logging_thread = Thread(
            target=Docker._logging,
            args=(self, logger, log_writer),
        )
        logging_thread.start()
        logging_thread.join()

        result = self._docker_api_client.exec_inspect(exec["Id"])

        return result

    def _logging(self, generator: Generator, log_writer: IO) -> None:
        """Logging thread for Client container

        Args:
            generator (string generator): Log stream.
        """
        try:
            while True:
                log = next(generator)
                txt = log.decode("utf-8")
                log_writer.write(txt)
                log_writer.flush()
        except StopIteration:
            pass
