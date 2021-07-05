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
from enum import Enum, auto

from model_navigator.core import Container
from model_navigator.framework import Framework, PyTorch, TensorFlow1, TensorFlow2


class TRITON_LOAD_MODE(Enum):
    POLL_ONCE = auto()
    POLL_PERIODICALLY = auto()
    EXPLICIT = auto()
    NONE = auto()


class TritonServer(Container):
    image = "nvcr.io/nvidia/tritonserver"
    tag = "py3"

    @staticmethod
    def library_path(framework: Framework):
        paths = {
            PyTorch.name: "/opt/tritonserver/lib/pytorch",
            TensorFlow1.name: "/opt/tritonserver/lib/tensorflow",
            TensorFlow2.name: "/opt/tritonserver/lib/tensorflow",
        }

        return paths[framework.name]

    @staticmethod
    def command(
        framework: Framework,
        repository_path: str,
        verbose: bool = False,
        strict_mode: bool = False,
        load_mode: TRITON_LOAD_MODE = TRITON_LOAD_MODE.EXPLICIT,
        metrics: bool = False,
    ):
        triton_command = f"tritonserver --model-store={repository_path}"
        if load_mode in [TRITON_LOAD_MODE.POLL_ONCE, TRITON_LOAD_MODE.POLL_PERIODICALLY]:
            triton_command += " --model-control-mode=poll"
        if load_mode == TRITON_LOAD_MODE.POLL_PERIODICALLY:
            triton_command += " --repository- poll-secs=5"
        if load_mode == TRITON_LOAD_MODE.EXPLICIT:
            triton_command += " --model-control-mode=explicit"

        if verbose:
            triton_command += " --log-verbose=1"

        if not strict_mode:
            triton_command += " --strict-model-config=false"

        if not metrics:
            triton_command += " --allow-metrics=false --allow-gpu-metrics=false"

        if isinstance(framework, (TensorFlow1, TensorFlow2)):
            version = 1 if framework == TensorFlow1 else 2
            triton_command += f" --backend-config=tensorflow,version={version}"

        return triton_command

    @staticmethod
    def api_method(key):
        methods = {
            "livenessProbe": "/v2/health/live",
            "readinessProbe": "/v2/health/ready",
        }

        return methods[key]
