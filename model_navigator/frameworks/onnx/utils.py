# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""ONNX utils."""

import pathlib
from typing import List, Tuple

import numpy as np

ONNX_RT_TYPE_TO_NP = {
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(int8)": np.int8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": bool,
    "tensor(string)": str,
}


def get_onnx_io_names(onnx_path: pathlib.Path) -> Tuple[List, List]:
    """Get input and output metadata from ONNX model."""
    import onnx

    model = onnx.load_model(onnx_path.as_posix())

    input_names = [input.name for input in model.graph.input]
    output_names = [output.name for output in model.graph.output]
    return input_names, output_names
