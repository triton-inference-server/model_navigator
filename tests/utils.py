# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import os
import pathlib
import select
import subprocess
from typing import Optional

from loguru import logger

from model_navigator.commands.performance.nvml_handler import NvmlHandler


def gpu_count() -> int:
    with NvmlHandler() as nvm_handler:
        return nvm_handler.gpu_count


def exec_command(cmd, workspace=None, name=None, shell=False) -> Optional[int]:
    returncode = None
    _output = []
    process = None

    logger.info(f"Command: {cmd}")
    logger.info(f"Current working directory: {workspace}")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")  # to not buffer logs
    try:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, cwd=workspace, shell=shell
        ) as process:
            process = process
            while process.poll() is None and process.returncode is None:
                _read_outputs(process, _output)
    finally:
        if process:
            _read_outputs(process, _output)
            logger.info(f"{name} process finished with {process.returncode}")
            returncode = process.returncode
        else:
            raise RuntimeError(f"Unable to execute command {cmd}")

    return returncode


def _read_outputs(_process, _outputs):
    try:
        (rds, _, _) = select.select([_process.stdout, _process.stderr], [], [], 1)
    except ValueError:  # when selecting on closed files
        rds = []
    while rds and _process.poll() is None:
        for rd in rds:
            line = rd.readline().decode("utf-8").rstrip()
            if line:
                logger.info(line)
                _outputs.append(line)
        try:
            (rds, _, _) = select.select([_process.stdout, _process.stderr], [], [], 1)
        except ValueError:  # when selecting on closed files
            rds = []


def get_assets_path():
    return pathlib.Path(__file__).parent / "assets"


def onnx_to_trt(onnx_path, trt_output_path):
    exec_command(
        f"trtexec --onnx={onnx_path} --saveEngine={trt_output_path}",
        workspace=os.path.dirname(onnx_path),
        name="ONNX to TRT",
        shell=True,
    )
