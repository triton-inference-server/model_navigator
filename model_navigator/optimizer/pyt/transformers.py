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
import logging

import sh

from model_navigator import Format
from model_navigator.model_navigator_exceptions import ModelNavigatorOptimizerException
from model_navigator.optimizer.utils import execute_sh_command, prepare_log_header

LOGGER = logging.getLogger("pyt.transformers")


def ts2onnx(input_path, opset, output_path, shapes, value_ranges, verbose):
    ts2onnx_args = [
        "-mmodel_navigator.optimizer.pyt.ts2onnx",
        input_path,
        output_path,
        "--opset-version",
        opset,
    ]
    if value_ranges:
        value_ranges = dict(value_ranges)
        ts2onnx_args += [
            "--value-ranges",
        ] + [f"{name}:{min_value},{max_value}" for name, (min_value, max_value) in value_ranges.items()]
    if shapes:
        ts2onnx_args += [
            "--shapes",
        ] + [f"{spec.name}:{','.join(map(str, spec.shape))}" for spec in shapes]
    if verbose:
        ts2onnx_args += ["-v"]

    log_path = output_path.parent / f"{output_path.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    python = sh.Command("python")
    try:
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TS_TRACE, Format.ONNX)
            execute_sh_command(python.bake(*ts2onnx_args), log_file=log_file, verbose=verbose)
        LOGGER.info("Optimization succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"Optimization failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorOptimizerException(e)
