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
import locale
import logging
import os
import platform
import re
from pathlib import Path
from subprocess import CalledProcessError
from typing import Dict, List

import cpuinfo
import psutil
import yaml

from model_navigator.utils import Workspace

LOGGER = logging.getLogger(__name__)

PACKAGES = [
    "tensorflow",
    "torch",
    "torch-tensorrt",
    "torchtext",
    "torchvision",
    "tensorrt",
    "tritonclient",
    "triton-model-analyzer",
    "xgboost",
    "tensorboard",
    "tensorboard-data-server",
    "tensorboard-plugin-wit",
    "polygraphy",
    "onnx",
    "onnxruntime-gpu",
    "onnx_graphsurgeon",
    "numpy",
    "tf2onnx",
]

LIBRARIES = {
    "NPP_VERSION",
    "DALI_BUILD",
    "CUSOLVER_VERSION",
    "CUBLAS_VERSION",
    "CUFFT_VERSION",
    "NCCL_VERSION",
    "CUSPARSE_VERSION",
    "OPENUCX_VERSION",
    "DRIVER_VERSION",
    "NSIGHT_SYSTEMS_VERSION",
    "TRT_VERSION",
    "CUDA_VERSION",
    "CURAND_VERSION",
    "DLPROF_VERSION",
    "OPENMPI_VERSION",
    "NVJPEG_VERSION",
    "CUDNN_VERSION",
    "NSIGHT_COMPUTE_VERSION",
    "DALI_VERSION",
    "NVIDIA_BUILD_ID",
    "CUDA_DRIVER_VERSION",
    "TRTOSS_VERSION",
    "NVIDIA_TENSORFLOW_VERSION",
    "TENSORFLOW_VERSION",
    "NVIDIA_PYTORCH_VERSION",
    "PYTORCH_VERSION",
    "TRITON_SERVER_VERSION",
    "NVIDIA_TRITON_SERVER_VERSION",
    "CONTAINER_VERSION",
}


def _command_runner(command):
    import subprocess

    return (
        subprocess.run(command, check=True, start_new_session=True, stdout=subprocess.PIPE)
        .stdout.decode(locale.getpreferredencoding())
        .strip()
    )


def _search(input: str, regex: str):
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def _split_to_dict(lines: List[str], separator: str):
    output = dict(map(lambda line: line.split(separator), lines))
    output = {k: v.strip() for k, v in output.items()}
    return output


def _get_packages():
    return _split_to_dict(
        _command_runner(["pip", "list", "--format=freeze", "--disable-pip-version-check"]).splitlines(), "=="
    )


def _remove(input: str, regex: str):
    return re.sub(regex, "", input).strip()


def get_git_info(disable_git_info: bool):
    git_info = {
        "repository": None,
        "commit": None,
        "author": None,
        "email": None,
    }
    if disable_git_info:
        return git_info
    try:
        git_info = {
            "repository": _command_runner(command=["git", "config", "--get", "remote.origin.url"]),
            "commit": _command_runner(command=["git", "log", "-1", "--pretty=format:%H"]),
            "author": _command_runner(command=["git", "log", "-1", "--pretty=format:%an"]),
            "email": _command_runner(command=["git", "log", "-1", "--pretty=format:%ae"]),
        }
    except CalledProcessError as e:
        LOGGER.warning(f"Unable to get git info: {e}")

    return git_info


def get_os_info():
    os_details = {
        "name": os.name,
        "platform": platform.system(),
        "release": platform.release(),
    }
    return os_details


def get_cpu_info():
    cpu_details = {
        "name": cpuinfo.get_cpu_info()["brand_raw"],
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "min_frequency": psutil.cpu_freq().min,
        "max_frequency": psutil.cpu_freq().max,
    }
    return cpu_details


def get_gpu_info():

    try:
        data = _command_runner(
            command=["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.max_limit", "--format=csv"]
        )
        lines = data.split(sep="\n")
        device_details = lines[1].split(",")

        cuda_version = None
        data = _command_runner(command=["nvidia-smi", "--query"])
        lines = data.split(sep="\n")
        for line in lines:
            if line.startswith("CUDA Version"):
                cuda_version = line.split(":")[1].strip()
                break

        gpu_details = {
            "name": device_details[0].strip(),
            "driver_version": device_details[1].strip(),
            "memory": device_details[2].strip(),
            "tdp": device_details[3].strip(),
            "cuda_version": cuda_version,
        }
    except (FileNotFoundError, CalledProcessError) as e:
        LOGGER.debug(str(e))
        gpu_details = {
            "name": "n/a",
            "driver_version": "n/a",
            "memory": "n/a",
            "tdp": "n/a",
            "cuda_version": "n/a",
        }

    return gpu_details


def get_env():
    packages = _get_packages()

    os_details = get_os_info()
    cpu_details = get_cpu_info()
    gpu_details = get_gpu_info()

    env = {
        "cpu": cpu_details,
        "memory": psutil._common.bytes2human(psutil.virtual_memory().total),
        "gpu": gpu_details,
        "os": os_details,
        "python_version": platform.python_version(),
        "python_packages": {k: v for k, v in packages.items() if k in PACKAGES},
        "libraries": {k: v for k, v in os.environ.items() if k in LIBRARIES},
    }
    return env


class EnvironmentStore:
    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def dump(self, stage, environment: Dict) -> Path:
        status_path: Path = self.get_path(stage)
        LOGGER.debug(f"Saving environment info for {stage} stage into {status_path}")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("w") as results_file:
            yaml.dump(environment, results_file)
        return status_path

    def load(self, stage):
        status_path: Path = self.get_path(stage)
        if not status_path.exists():
            LOGGER.warning(f"No environment found for {stage}")
            return {}

        with status_path.open("r") as results_file:
            environment = yaml.safe_load(results_file)

        return environment

    def get_path(self, stage: str) -> Path:
        return self._workspace.path / f"{stage}_environment.yaml"
