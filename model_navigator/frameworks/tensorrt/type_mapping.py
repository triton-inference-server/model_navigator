# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Utilities for mapping dtypes between TensorRT and frameworks."""

from packaging.version import Version

from model_navigator.utils import module

torch = module.lazy_import("torch")
trt = module.lazy_import("tensorrt")


def _get_trt_to_torch_dtype_dict():
    trt_to_torch_dtype = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }

    if Version(trt.__version__) >= Version("9.0"):
        trt_to_torch_dtype[trt.DataType.INT64] = torch.int64

    if Version(trt.__version__) >= Version("9.2"):
        trt_to_torch_dtype[trt.DataType.BF16] = torch.bfloat16

    return trt_to_torch_dtype


def trt_to_torch_dtype(trt_dtype: "trt.DataType") -> "torch.dtype":
    """Cast TensorRT DataType to torch.dtype.

    Args:
        trt_dtype (trt.DataType): TensorRT DataType

    Returns:
        torch.dtype: PyTorch dtype
    """
    return _get_trt_to_torch_dtype_dict()[trt_dtype]


def _get_torch_to_trt_dtype_dict():
    return {v: k for k, v in _get_trt_to_torch_dtype_dict().items()}


def torch_to_trt_dtype(torch_dtype: "torch.dtype") -> "trt.DataType":
    """Cast torch.dtype to TensorRT DataType.

    Args:
        torch_dtype (torch.dtype): PyTorch dtype

    Returns:
        trt.DataType: TensorRT DataType
    """
    return _get_torch_to_trt_dtype_dict()[torch_dtype]
