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

from pathlib import Path
from typing import Union

from model_navigator.model import Format
from model_navigator.utils.formats.onnx import ONNXUtils
from model_navigator.utils.formats.pytorch import TorchScriptUtils
from model_navigator.utils.formats.tensorflow import TensorFlowSavedModelUtils
from model_navigator.utils.formats.tensorrt import TensorRTUtils

SUFFIX2FORMAT = {
    ".savedmodel": Format.TF_SAVEDMODEL,
    ".plan": Format.TENSORRT,
    ".onnx": Format.ONNX,
    ".pt": Format.TORCHSCRIPT,
}
FORMAT2SUFFIX = {format_: suffix for suffix, format_ in SUFFIX2FORMAT.items()}


def guess_format(model_path: Union[str, Path]):
    model_path = Path(model_path)
    suffix = model_path.suffix

    try:
        file_format = SUFFIX2FORMAT[suffix]
    except KeyError:
        file_format = None
    return file_format


FORMAT2ADAPTER = {
    Format.ONNX: ONNXUtils,
    Format.TORCHSCRIPT: TorchScriptUtils,
    Format.TENSORRT: TensorRTUtils,
    Format.TF_SAVEDMODEL: TensorFlowSavedModelUtils,
}
