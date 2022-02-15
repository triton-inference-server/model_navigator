# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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
import pathlib
import typing

import docker

from model_navigator.triton.client import TritonClient
from model_navigator.triton.server.exceptions import TritonServerException
from model_navigator.triton.server.server import TritonServer
from model_navigator.utils.docker import DockerContainer, DockerImage

LOCAL_HTTP_PORT = 8000
LOCAL_GRPC_PORT = 8001
LOCAL_METRICS_PORT = 8002

LOGGER = logging.getLogger(__name__)


class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """

    def __init__(self, *, image, config, gpus, path):
        """
        Parameters
        ----------
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus : list
            list of GPUs to be used
        """

        super().__init__(config=config, gpus=gpus, path=path)
        self._docker_image = DockerImage(image)
        self._docker_container: typing.Optional[DockerContainer] = None
        self._server_logs_buffer = io.StringIO()
        self._server_logs_gen = None

        assert self._server_config["model-repository"], "Triton Server requires --model-repository argument to be set."

    def start(self):
        """
        Starts the tritonserver docker container using docker-py
        """
        devices = self._get_devices()
        mount_as_volumes = [pathlib.Path(self._server_config["model-repository"])]

        # Map ports, use config values but set to server defaults if not
        # specified
        ports = self.get_ports()
        ports = {port_number: port_number for port_number in ports.values()}

        try:
            self._docker_container: DockerContainer = self._docker_image.run_container(
                devices=devices,
                mount_as_volumes=mount_as_volumes,
                ports=ports,
            )
        except docker.errors.APIError as error:
            if error.explanation.find("port is already allocated") != -1:
                raise TritonServerException(
                    "One of the following port(s) are already allocated: "
                    f"{', '.join(map(str, ports.values()))}.\n"
                    "Change the Triton server ports using"
                    " --triton-http-endpoint, --triton-grpc-endpoint,"
                    " and --triton-metrics-endpoint flags."
                )
            else:
                raise error

        # Run the command in the container
        cmd = self._server_path + " " + self._server_config.to_cli_string()
        self._server_logs_gen = self._docker_container.run_cmd(cmd=cmd, detached=True)

    def _get_devices(self):
        if len(self._gpus) == 1 and self._gpus[0] == "all":
            devices = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        else:
            devices = [docker.types.DeviceRequest(device_ids=self._gpus, capabilities=[["gpu"]])]
        return devices

    def stop(self):
        """
        Stops the tritonserver docker container
        and cleans up docker client
        """

        if self._docker_container:
            LOGGER.debug(f"Stopping Triton Inference Server (container={self._docker_container.id}).")
            self._docker_container.kill()
            self._docker_container = None
        LOGGER.debug("Triton server stopped")

    def is_alive(self):
        # TODO: check if tritonserver process is running inside container
        return self._docker_container is not None

    def logs(self):
        """
        Retrieves the Triton server's stdout
        as a str
        """
        if self._server_logs_gen:
            for chunk in self._server_logs_gen:
                self._server_logs_buffer.write(chunk.decode("utf-8"))

        return self._server_logs_buffer.getvalue()

    def get_ports(self):
        return {
            "http": self._server_config["http-port"] or 8000,
            "grpc": self._server_config["grpc-port"] or 8001,
            "metrics": self._server_config["metrics-port"] or 8002,
        }

    def create_grpc_client(self):
        port = self.get_ports()["grpc"]
        return TritonClient(f"grpc://{self._docker_container.ip_address}:{port}")

    def create_http_client(self):
        port = self.get_ports()["http"]
        container = DockerContainer(self._docker_container.id)
        return TritonClient(f"http://{container.ip_address}:{port}")
