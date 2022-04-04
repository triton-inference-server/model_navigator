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

from model_navigator.converter.config import TensorRTConversionConfig
from model_navigator.converter.dataloader import Dataloader
from model_navigator.converter.utils import navigator_subprocess, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException
from model_navigator.model import Format, ModelSignatureConfig

LOGGER = logging.getLogger(__name__)


def ts2torchtrt(
    input_path,
    output_path,
    *,
    log_path,
    dataloader: Dataloader,
    signature_config: Optional[ModelSignatureConfig],
    tensorrt_config: TensorRTConversionConfig,
    max_batch_size: int,
    verbose: bool = False,
):
    LOGGER.info("%s command started.", __name__)

    with log_path.open("w") as log_file:
        prepare_log_header(log_file, Format.TORCHSCRIPT, Format.TORCH_TRT)
        try:
            with navigator_subprocess(log_file=log_file, verbose=verbose) as navigator:
                ts2trt = navigator.module("model_navigator.converter.torch_tensorrt.ts2trt")
                ts2trt.compile(
                    **{
                        "input_model": input_path,
                        "output_path": output_path,
                        "log_path": log_path,
                        "signature_config": signature_config,
                        "dataloader": dataloader,
                        "tensorrt_config": tensorrt_config,
                        "max_batch_size": max_batch_size,
                    }
                )
        except Exception as e:
            LOGGER.warning(f"torch-tensorrt conversion failed. Details can be found in logfile: {log_path}")
            raise ModelNavigatorConverterCommandException(message=str(e), log_path=log_path)
