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
from typing import Optional

import sh

from model_navigator.cli.spec import serialize_shapes
from model_navigator.converter import DatasetProfileConfig
from model_navigator.converter.config import ConversionConfig
from model_navigator.converter.utils import execute_sh_command, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException
from model_navigator.model import Format, ModelSignatureConfig

LOGGER = logging.getLogger(__name__)


def _get_shapes(dataset_profile):
    if dataset_profile is None:
        return []

    shapes = []
    if dataset_profile.min_shapes:
        for shape in serialize_shapes(None, value=dataset_profile.min_shapes):
            shapes += ["--trt-min-shapes", shape]
    if dataset_profile.max_shapes:
        for shape in serialize_shapes(None, value=dataset_profile.max_shapes):
            shapes += ["--trt-max-shapes", shape]
    if dataset_profile.opt_shapes:
        for shape in serialize_shapes(None, value=dataset_profile.opt_shapes):
            shapes += ["--trt-opt-shapes", shape]
    return shapes


def ts2torchtrt(
    input_path,
    output_path,
    *,
    log_path,
    dataset_profile: Optional[DatasetProfileConfig],
    signature_config: Optional[ModelSignatureConfig],
    conversion_config: ConversionConfig,
    max_workspace_size: int,
    max_batch_size: int,
    verbose: bool = False,
):
    LOGGER.info("%s command started.", __name__)
    with log_path.open("w") as log_file:
        prepare_log_header(log_file, Format.TORCHSCRIPT, Format.TORCH_TRT)
        shapes = _get_shapes(dataset_profile)
        try:
            execute_sh_command(
                sh.python3.bake(
                    "-mmodel_navigator.converter.torch_tensorrt.ts2trt",
                    *shapes,
                    **{
                        "input_model": input_path,
                        "output_path": output_path,
                        "log_path": log_path,
                        "precision": conversion_config.tensorrt_precision.value,
                        "precision_mode": conversion_config.tensorrt_precision_mode.value,
                        "tensorrt_sparse_weights": conversion_config.tensorrt_sparse_weights,
                        "tensorrt_strict_types": conversion_config.tensorrt_strict_types,
                        "max_batch_size": max_batch_size,
                        "max_workspace_size": max_workspace_size,
                        "verbose": bool(verbose),
                    },
                ),
                log_file=log_file,
            )
        except sh.ErrorReturnCode as e:
            LOGGER.warning(f"torch-tensorrt conversion failed. Details can be found in logfile: {log_path}")
            raise ModelNavigatorConverterCommandException(message=e.stdout.decode("utf-8"), log_path=log_path)
