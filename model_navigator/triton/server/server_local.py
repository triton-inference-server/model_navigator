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
import os
import traceback

import sh

from model_navigator.triton.client import TritonClient
from model_navigator.triton.server.server import TritonServer

SERVER_OUTPUT_TIMEOUT_SECS = 30
LOGGER = logging.getLogger(__name__)


def handle_exit(cmd, success, exit_code):
    if not success:
        LOGGER.warning("Triton Inference Server exited with failure. Please wait.")
        LOGGER.debug(f"Triton Inference Server exit code {exit_code}")
    else:
        LOGGER.debug("Triton Inference Server stopped")


class TritonServerLocal(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    tritonserver locally as as subprocess.
    """

    def __init__(self, *, path, config, gpus):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the tritonserver executable
        config : TritonServerConfig
            the config object containing arguments for this server instance
        """
        super().__init__(config=config, gpus=gpus, path=path)
        self._tritonserver_running_cmd = None
        self._tritonserver_logs = ""

        assert self._server_config["model-repository"], "Triton Server requires --model-repository argument to be set."

    def start(self):
        """
        Starts the tritonserver locally
        """

        if self.is_alive():
            raise RuntimeError(
                f"You have to stop previously started tritonserver process first "
                f"pid={self._tritonserver_running_cmd.pid}"
            )
        else:
            env = self._get_env()

            tritonserver_cmd, *rest = self._server_path.split(" ", 1)
            tritonserver_cmd = sh.Command(tritonserver_cmd)
            tritonserver_cmd = tritonserver_cmd.bake(*rest)

            tritonserver_args = self._server_config.to_cli_string().replace("=", " ").split()

            self._tritonserver_logs = ""
            self._tritonserver_running_cmd = tritonserver_cmd(
                *tritonserver_args,
                _env=env,
                _err_to_out=True,
                _out=self._record_logs,
                _bg=True,
                _bg_exc=False,
                _done=handle_exit,
            )

    def _record_logs(self, line):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        self._tritonserver_logs += line

    def _get_env(self):
        env = None
        if self._gpus:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self._gpus)

        return env

    def stop(self):
        """
        Stops the running tritonserver
        """

        if self.is_alive():
            self._tritonserver_running_cmd.process.terminate()
            try:
                self._tritonserver_running_cmd.wait(timeout=SERVER_OUTPUT_TIMEOUT_SECS)
            except Exception:
                LOGGER.debug("Timeout waiting for server. Trying to kill process.")
                message = traceback.format_exc()
                LOGGER.debug(f"Error message: \n{message}")
                try:
                    self._tritonserver_running_cmd.process.kill()
                    self._tritonserver_running_cmd.wait(timeout=SERVER_OUTPUT_TIMEOUT_SECS)
                except Exception:
                    LOGGER.debug(f"Could not kill triton server pid={self._tritonserver_running_cmd.pid}")
                    message = traceback.format_exc()
                    LOGGER.debug(f"Error message: \n{message}")

    def is_alive(self):
        return self._tritonserver_running_cmd is not None and self._tritonserver_running_cmd.is_alive()

    def logs(self):
        return self._tritonserver_logs

    def get_ports(self):
        return {
            "http": self._server_config["http-port"] or 8000,
            "grpc": self._server_config["grpc-port"] or 8001,
            "metrics": self._server_config["metrics-port"] or 8002,
        }

    def create_grpc_client(self):
        triton_ports = self.get_ports()
        return TritonClient(f"grpc://127.0.0.1:{triton_ports['grpc']}")

    def create_http_client(self):
        triton_ports = self.get_ports()
        return TritonClient(f"grpc://127.0.0.1:{triton_ports['grpc']}")
