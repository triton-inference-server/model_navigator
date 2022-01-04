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
from typing import Optional

import sh

from model_navigator.cli.spec import serialize_shapes, serialize_value_ranges
from model_navigator.converter import DatasetProfileConfig
from model_navigator.converter.config import TensorRTPrecision
from model_navigator.converter.utils import execute_sh_command, prepare_log_header
from model_navigator.core import DEFAULT_TENSORRT_MAX_WORKSPACE_SIZE
from model_navigator.exceptions import ModelNavigatorConverterCommandException
from model_navigator.model import Format

LOGGER = logging.getLogger("tf2tftrt.transformers")


def tf2tftrt(
    input_path,
    output_path,
    *,
    log_path,
    precision: Optional[TensorRTPrecision],
    dataset_profile: Optional[DatasetProfileConfig],
    max_workspace_size: Optional[int],
    max_batch_size: int,
    verbose: bool = False,
):
    LOGGER.info("tf2tftrt command started.")
    python = sh.Command("python")
    args = []
    if dataset_profile:
        if dataset_profile.max_shapes:
            args += [
                "--trt-max-shapes",
                *serialize_shapes(None, value=dataset_profile.max_shapes),
            ]
        if dataset_profile.min_shapes:
            args += [
                "--trt-min-shapes",
                *serialize_shapes(None, value=dataset_profile.min_shapes),
            ]
        if dataset_profile.value_ranges:
            args += ["--value-ranges", *serialize_value_ranges(None, value=dataset_profile.value_ranges)]

    if max_batch_size:
        args += [
            "--max-batch-size",
            max_batch_size,
        ]

    tf2tf_trt_args = [
        "-mmodel_navigator.converter.tf_trt.tf_trt_convert",
        "--input-path",
        input_path,
        "--output-path",
        output_path,
        *args,
        "--precision",
        precision.value if precision else "FP32",
        "--max-workspace-size",
        max_workspace_size or DEFAULT_TENSORRT_MAX_WORKSPACE_SIZE,
    ]
    if verbose:
        tf2tf_trt_args += [
            "--verbose",
        ]

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TF_SAVEDMODEL, Format.TF_TRT)
            execute_sh_command(python.bake(*tf2tf_trt_args), log_file=log_file, verbose=verbose)
        LOGGER.info("tf2tftrt command succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"tf2tftrt conversion failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=e.stdout.decode("utf-8"), log_path=log_path)
