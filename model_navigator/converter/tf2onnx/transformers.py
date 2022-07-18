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

from model_navigator.converter.utils import execute_sh_command, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException
from model_navigator.model import Format

LOGGER = logging.getLogger("tf2onnx.transformers")


def tfopt(input_path, output_path, *, log_path, verbose: bool = False):
    import sh

    python = sh.Command("python")
    args = [
        "-mmodel_navigator.converter.tf2onnx.tf_opt",
        "--saved-model",
        input_path,
        "--output",
        output_path,
    ]
    if verbose:
        args += [
            "--verbose",
        ]
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            execute_sh_command(python.bake(*args), log_file=log_file, verbose=verbose)
        LOGGER.info("tfopt command succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"tf SavedModel optimization failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=e.stdout.decode("utf-8"), log_path=log_path)


def tf2onnx(input_path, output_path, *, log_path, opset: int, verbose: bool = False):
    LOGGER.info("tf2onnx command started.")
    tf2onnx_args = [
        "-mtf2onnx.convert",
        "--saved-model",
        input_path,
        "--output",
        output_path,
        "--opset",
        opset,
    ]
    LARGE_MODEL_THRESHOLD = 2 * (2**30)  # 2GB
    total_files_size = sum(p.stat().st_size for p in input_path.rglob("*") if p.is_file())
    is_large_file = total_files_size > LARGE_MODEL_THRESHOLD
    if is_large_file:
        tf2onnx_args += ["--large_model"]
    rename_idx_args = [
        "-mmodel_navigator.converter.tf2onnx.remove_idx_from_inputs",
        output_path,
        output_path,
    ]
    import sh

    if verbose:
        tf2onnx_args += ["-vvv"]
        rename_idx_args += ["-vvv"]
    python = sh.Command("python")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TF_SAVEDMODEL, Format.ONNX)
            execute_sh_command(python.bake(*tf2onnx_args), log_file=log_file, verbose=verbose)
            execute_sh_command(python.bake(*rename_idx_args), log_file=log_file, verbose=verbose)
        LOGGER.info("tf2onnx command succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"tf2onnx conversion failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=e.stdout.decode("utf-8"), log_path=log_path)
