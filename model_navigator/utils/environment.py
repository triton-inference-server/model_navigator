# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Collect information about current system and environment."""

import locale
import logging
import os
import platform
import re
from subprocess import CalledProcessError
from typing import Dict, List, Optional, Union

import cpuinfo
import psutil

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


def get_os_info():
    """Collect information about the OS."""
    os_details = {
        "name": os.name,
        "platform": platform.system(),
        "release": platform.release(),
    }
    return os_details


def get_cpu_info():
    """Collect information about CPU available in the system."""
    cpu_details = {
        "name": cpuinfo.get_cpu_info().get("brand_raw", "n/a"),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "min_frequency": psutil.cpu_freq().min,
        "max_frequency": psutil.cpu_freq().max,
    }
    return cpu_details


def get_gpu_info() -> Dict:
    """Collect information about NVIDIA GPU available in the system.

    Returns:
        Dictionary with GPU information
    """
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


def get_env() -> Dict:
    """Collect information from current environment.

    This method gather information about Python packages, environment variables, OS version as well as
    CPU and GPU information.

    Returns:
        Dictionary with details about current working environment
    """
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


def _command_runner(command: Union[str, List]) -> str:
    """Executed command as subprocess and collect information."""
    import subprocess

    return (
        subprocess.run(command, check=True, start_new_session=True, stdout=subprocess.PIPE)
        .stdout.decode(locale.getpreferredencoding())
        .strip()
    )


def _search(input: str, regex: str) -> Optional[str]:
    """Search for matching regular expression inside given string.

    Returns:
        Matched string or None if not matches found
    """
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def _split_to_dict(lines: List[str], separator: str):
    """Split the lines of string into a dictionary based on separator.

    First value from split is a key and second is value.

    Returns:
        Dictionary with split values.
    """
    output = dict(map(lambda line: line.split(separator), lines))
    output = {k: v.strip() for k, v in output.items()}
    return output


def _get_packages():
    """Collect Python packages installed in the system along with the version."""
    return _split_to_dict(
        _command_runner(["pip", "list", "--format=freeze", "--disable-pip-version-check"]).splitlines(), "=="
    )


def _remove(input: str, regex: str) -> str:
    """Format data based on regular expression.

    Returns:
        Formatted string
    """
    return re.sub(regex, "", input).strip()
