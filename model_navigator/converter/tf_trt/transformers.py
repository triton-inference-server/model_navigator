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

from model_navigator.converter.config import TensorRTConversionConfig
from model_navigator.converter.dataloader import Dataloader
from model_navigator.converter.utils import navigator_subprocess, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException
from model_navigator.model import Format

LOGGER = logging.getLogger("tf2tftrt.transformers")


def tf2tftrt(
    input_path,
    output_path,
    *,
    log_path,
    tensorrt_config: TensorRTConversionConfig,
    dataloader: Dataloader,
    max_batch_size: int,
    verbose: bool = False,
):
    LOGGER.info("tf2tftrt command started.")

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TF_SAVEDMODEL, Format.TF_TRT)
            with navigator_subprocess(log_file=log_file, verbose=verbose) as navigator:
                tf_trt_convert = navigator.module("model_navigator.converter.tf_trt.tf_trt_convert")
                tf_trt_convert.convert_tf2(
                    input_path=input_path,
                    output_path=output_path,
                    max_workspace_size=tensorrt_config.max_workspace_size,
                    precision=tensorrt_config.precision.value,
                    dataloader=dataloader,
                )
        LOGGER.info("tf2tftrt command succeed.")
    except Exception as e:
        LOGGER.warning(f"tf2tftrt conversion failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=str(e), log_path=log_path)
