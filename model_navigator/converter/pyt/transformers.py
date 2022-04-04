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

from model_navigator.converter.utils import navigator_subprocess, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException
from model_navigator.model import Format

LOGGER = logging.getLogger("pyt.transformers")


def ts2onnx(*, input_path, output_path, log_path, opset, dataloader, verbose):
    LOGGER.info("ts2onnx command started.")

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TORCHSCRIPT, Format.ONNX)
            with navigator_subprocess(log_file=log_file, verbose=verbose) as navigator:
                ts2onnx = navigator.module("model_navigator.converter.pyt.ts2onnx")
                ts2onnx.convert(input_path, output_path, opset, dataloader, verbose)
        LOGGER.info("ts2onnx command succeed.")
    except Exception as e:
        LOGGER.warning(f"ts2onnx conversion failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=str(e), log_path=log_path)
