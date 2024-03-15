# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""Internal configuration classes and methods."""

import enum
from typing import Any, Optional, Tuple, Type, Union

import numpy as np

from model_navigator.exceptions import ModelNavigatorWrongParameterError


class Backend(enum.Enum):
    """Define supported backends for which model store can be created."""

    ONNXRuntime = "onnxruntime"
    TensorRT = "tensorrt"
    PyTorch = "pytorch"
    TensorFlow = "tensorflow"
    Python = "python"


def expect_type(name: str, value: Any, expected_types: Type, optional=False):
    """Validate if value of parameter has expected type."""
    if not (isinstance(value, expected_types) or (value is None and optional)):
        raise ModelNavigatorWrongParameterError(f"`{name}` argument should be {expected_types}, but got {type(value)}.")


def is_dim_correct(dim: Any):
    """Validate if single shape element is valid."""
    # int equal to -1 or positive number
    return isinstance(dim, int) and (dim == -1 or dim > 0)


def is_shape_correct(name: str, value: Optional[Tuple], optional: bool = False):
    """Validate if of parameter is correct."""
    if not value and not optional:
        raise ModelNavigatorWrongParameterError(f"Empty {name} is not supported.")

    if not all(is_dim_correct(dim) for dim in value):
        raise ModelNavigatorWrongParameterError(
            f"{name} items should be integers equal to -1 or positive numbers. Got {value}."
        )


def cast_dtype(dtype: Union[np.dtype, Type[np.dtype]]):
    """Cast provided argument to np.dtype."""
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)

    return dtype
