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

from abc import ABC, abstractmethod
from typing import List


class TritonServer(ABC):
    """
    Defines the interface for the objects created by
    TritonServerFactory
    """

    def __init__(self, *, config, gpus, path):
        self._server_path = path
        self._server_config = config
        self._gpus = gpus

    @abstractmethod
    def start(self):
        """
        Starts the tritonserver
        """

    @abstractmethod
    def stop(self):
        """
        Stops and cleans up after the server
        """

    @abstractmethod
    def is_alive(self):
        pass

    @abstractmethod
    def logs(self):
        """
        Gets the server's stdout logs as a string
        """

    @abstractmethod
    def create_grpc_client(self):
        pass

    @abstractmethod
    def create_http_client(self):
        pass

    def set_gpus(self, gpus: List[str]):
        if self.is_alive():
            raise RuntimeError("Triton Inference Server is running thus could not change gpus")
        self._gpus = gpus
