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
from typing import Optional, Union

from dataclasses import dataclass
from pathlib import Path

from .config import ModelNavigatorBaseConfig
from .core import Accelerator, Format, Precision
from .model_navigator_exceptions import ModelNavigatorException

SUFFIX2FORMAT = {
    ".savedmodel": Format.TF_SAVEDMODEL,
    ".plan": Format.TRT,
    ".onnx": Format.ONNX,
    ".pt": Format.TS_SCRIPT,
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


class InputModel:
    name: str
    path: Path
    format: Format
    config: Optional[ModelNavigatorBaseConfig]

    def __init__(self, name: str, path: Union[str, Path], config: Optional[ModelNavigatorBaseConfig] = None):
        self.name = name
        self.path = Path(path)
        self.config = config

        model_format = guess_format(self.path)
        if not model_format:
            raise ModelNavigatorException(
                f"""Unsupported file type in {self.path}. """
                """Please provide file with one of the following file type: """
                f"""{", ".join(list(map(lambda ext: f"*{ext}", SUFFIX2FORMAT.keys())))}"""
            )

        self.format = model_format


@dataclass
class Model:
    base_name: str
    name: str
    format: Format
    max_batch_size: int
    precision: Precision
    accelerator: Accelerator
    capture_cuda_graph: int
    gpu_engine_count: int
    onnx_opset: int
    path: Path
    triton_path: Optional[Path] = None
    error_log: Optional[Path] = None
    result_path: Optional[Path] = None
