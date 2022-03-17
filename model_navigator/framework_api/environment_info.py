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
import os
import platform
import re
from subprocess import CalledProcessError
from typing import List

import psutil

from model_navigator.framework_api.logger import LOGGER

package_filter = [
    "tensorflow" "torch",
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
    return _split_to_dict(_command_runner(["pip", "list", "--format=freeze"]).splitlines(), "==")


def _remove(input: str, regex: str):
    return re.sub(regex, "", input).strip()


def get_env():
    packages = _get_packages()

    os_details = {
        "name": os.name,
        "platform": platform.system(),
        "release": platform.release(),
    }

    env = {
        "cpu": platform.processor(),
        "memory": psutil._common.bytes2human(psutil.virtual_memory().total),
        "gpu": _split_to_dict(
            _remove(_command_runner(command=["nvidia-smi", "-L"]), regex=r"\(UUID: .+?\)").splitlines(), ":"
        ),
        "driver_version": _search(input=_command_runner(command=["nvidia-smi"]), regex=r"Driver Version: (.*?) "),
        "os": os_details,
        "python_version": platform.python_version(),
        "python_packages": {k: v for k, v in packages.items() if k in package_filter},
    }
    return env


def get_git_info():
    try:
        git_info = {
            "repository": _command_runner(command=["git", "config", "--get", "remote.origin.url"]),
            "commit": _command_runner(command=["git", "log", "--pretty=format:%H", "HEAD^..HEAD"]),
            "author": _command_runner(command=["git", "log", "--pretty=format:%an", "HEAD^..HEAD"]),
            "email": _command_runner(command=["git", "log", "--pretty=format:%ae", "HEAD^..HEAD"]),
        }
    except CalledProcessError as e:
        LOGGER.warning(f"Unable to get git info: {e}")
        git_info = {}
    return git_info
