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

LOGGER = logging.getLogger("tf2onnx.transformers")


def tfopt(input_path, output_path, *, verbose: bool = False):
    python = sh.Command("python")
    args = [
        "-mmodel_navigator.optimizer.tf2onnx.tf_opt",
        "--saved-model",
        input_path,
        "--output",
        output_path,
    ]
    if verbose:
        args += [
            "--verbose",
        ]
    log_path = output_path.parent / f"{output_path.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log_file:
            execute_sh_command(python.bake(*args), log_file=log_file, verbose=verbose)
        LOGGER.info("Optimization succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"Optimization failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorOptimizerException(e)


def tf2onnx(input_path, output_path, *, opset: int, verbose: bool = False):
    tf2onnx_args = [
        "-mtf2onnx.convert",
        "--saved-model",
        input_path,
        "--output",
        output_path,
        "--opset",
        opset,
    ]
    LARGE_MODEL_THRESHOLD = 2 * (2 ** 30)  # 2GB
    total_files_size = sum([p.stat().st_size for p in input_path.rglob("*") if p.is_file()])
    is_large_file = total_files_size > LARGE_MODEL_THRESHOLD
    if is_large_file:
        tf2onnx_args += ["--large_model_flags"]
    rename_idx_args = [
        "-mmodel_navigator.optimizer.tf2onnx.remove_idx_from_inputs",
        output_path,
        output_path,
    ]
    if verbose:
        tf2onnx_args += ["-vvv"]
        rename_idx_args += ["-vvv"]
    python = sh.Command("python")
    log_path = output_path.parent / f"{output_path.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TF_SAVEDMODEL, Format.ONNX)
            execute_sh_command(python.bake(*tf2onnx_args), log_file=log_file, verbose=verbose)
            execute_sh_command(python.bake(*rename_idx_args), log_file=log_file, verbose=verbose)
        LOGGER.info("Optimization succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"Optimization failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorOptimizerException(e)
