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
import logging
from typing import Dict, Tuple

import sh

from model_navigator.cli.spec import serialize_shapes, serialize_value_ranges
from model_navigator.converter.utils import execute_sh_command, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException
from model_navigator.model import Format

LOGGER = logging.getLogger("pyt.transformers")


def ts2onnx(
    *, input_path, output_path, log_path, opset, shapes: Dict[str, Tuple], value_ranges: Dict[str, Tuple], verbose
):
    LOGGER.info("ts2onnx command started.")
    ts2onnx_args = [
        "-mmodel_navigator.converter.pyt.ts2onnx",
        input_path,
        output_path,
        "--opset-version",
        opset,
    ]
    if value_ranges:
        ts2onnx_args += ["--value-ranges"] + serialize_value_ranges(None, value_ranges)
    if shapes:
        ts2onnx_args += ["--shapes"] + serialize_shapes(None, value=shapes)
    if verbose:
        ts2onnx_args += ["-v"]

    python = sh.Command("python")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TORCHSCRIPT, Format.ONNX)
            execute_sh_command(python.bake(*ts2onnx_args), log_file=log_file, verbose=verbose)
        LOGGER.info("ts2onnx command succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"ts2onnx conversion failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=e.stdout.decode("utf-8"), log_path=log_path)
