# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import docker

from model_navigator.triton.client import TritonClient
from model_navigator.triton.server.exceptions import TritonServerException
from model_navigator.triton.server.server import TritonServer
from model_navigator.utils.docker import DockerContainer

LOCAL_HTTP_PORT = 8000
LOCAL_GRPC_PORT = 8001
LOCAL_METRICS_PORT = 8002

logger = logging.getLogger(__name__)


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
        self._docker_client = docker.from_env()
        self._tritonserver_image = image
        self._tritonserver_container = None
        self._tritonserver_log_gen = None

        assert self._server_config["model-repository"], "Triton Server requires --model-repository argument to be set."

    def start(self):
        """
        Starts the tritonserver docker container using docker-py
        """
        logger.debug("Starting triton server.")

        if len(self._gpus) == 1 and self._gpus[0] == "all":
            devices = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        else:
            devices = [docker.types.DeviceRequest(device_ids=self._gpus, capabilities=[["gpu"]])]

        # Mount required directories
        volumes = {
            self._server_config["model-repository"]: {"bind": self._server_config["model-repository"], "mode": "ro"}
        }

        # Map ports, use config values but set to server defaults if not
        # specified
        triton_ports = self.get_ports()

        ports = {
            triton_ports["http"]: triton_ports["http"],
            triton_ports["grpc"]: triton_ports["grpc"],
            triton_ports["metrics"]: triton_ports["metrics"],
        }

        try:
            # Run the docker container
            logger.debug(
                f"Starting docker container with {self._tritonserver_image} device_requests={devices} "
                f"volumes={volumes} ports={ports}"
            )
            self._tritonserver_container = self._docker_client.containers.run(
                image=self._tritonserver_image,
                device_requests=devices,
                volumes=volumes,
                ports=ports,
                publish_all_ports=True,
                tty=True,
                stdin_open=True,
                detach=True,
                ipc_mode="host",
            )
        except docker.errors.APIError as error:
            if error.explanation.find("port is already allocated") != -1:
                raise TritonServerException(
                    "One of the following port(s) are already allocated: "
                    f"{', '.join(map(str, triton_ports.values()))}.\n"
                    "Change the Triton server ports using"
                    " --triton-http-endpoint, --triton-grpc-endpoint,"
                    " and --triton-metrics-endpoint flags."
                )
            else:
                raise error

        # Run the command in the container
        cmd = self._server_path + " " + self._server_config.to_cli_string()

        logger.debug(f"Run command {cmd} in docker container id={self._tritonserver_container.id}")
        _, self._tritonserver_log_gen = self._tritonserver_container.exec_run(cmd=cmd, stream=True)

    def stop(self):
        """
        Stops the tritonserver docker container
        and cleans up docker client
        """

        logger.debug("Stopping triton server.")

        if self.is_alive():
            logger.debug(f"Stopping Triton Server id={self._tritonserver_container.id}")
            self._tritonserver_container.stop()
            self._tritonserver_container.remove(force=True)

            self._tritonserver_container = None
            self._docker_client.close()

        logger.debug("Triton server stopped")

    def is_alive(self):
        # TODO: check if tritonserver process is running inside container
        return self._tritonserver_container is not None

    def logs(self):
        """
        Retrieves the Triton server's stdout
        as a str
        """

        return b"".join(list(self._tritonserver_log_gen)).decode("utf-8")

    def get_ports(self):
        return {
            "http": self._server_config["http-port"] or 8000,
            "grpc": self._server_config["grpc-port"] or 8001,
            "metrics": self._server_config["metrics-port"] or 8002,
        }

    def create_grpc_client(self):
        port = self.get_ports()["grpc"]
        container = DockerContainer(self._tritonserver_container.id)
        return TritonClient(f"grpc://{container.ip_address}:{port}")

    def create_http_client(self):
        port = self.get_ports()["http"]
        container = DockerContainer(self._tritonserver_container.id)
        return TritonClient(f"http://{container.ip_address}:{port}")
