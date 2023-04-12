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
"""TensorRT utils."""
import logging
from distutils.version import LooseVersion

import numpy as np

from model_navigator.api.config import ShapeTuple, TensorRTProfile
from model_navigator.utils import module
from model_navigator.utils.tensor import TensorSpec

LOGGER = logging.getLogger(__name__)
_TYPE_CASTS = {
    np.dtype(np.int64): np.dtype(np.int32),
    np.dtype(np.float64): np.dtype(np.float32),
    np.dtype(np.uint64): np.dtype(np.uint32),
}


def get_version():
    """Get TensorRT version."""
    trt = module.lazy_import("tensorrt")
    trt_version = LooseVersion(trt.__version__)
    return trt_version


def get_trt_profile_from_trt_dynamic_axes(trt_dynamic_axes):
    """Create TensorRT profile from dynamic axes."""
    trt_profile = TensorRTProfile()
    if trt_dynamic_axes is None:
        return trt_profile
    for name, axes in trt_dynamic_axes.items():
        if axes:
            trt_profile.add(name, *list(zip(*list(axes.values()))))
    return trt_profile


def cast_type(dtype: np.dtype) -> np.dtype:
    """Cast type and return new dtype."""
    if dtype in _TYPE_CASTS:
        return _TYPE_CASTS[dtype]

    return dtype


def cast_tensor(tensor: TensorSpec) -> TensorSpec:
    """Cast type and return new dtype."""
    if tensor.dtype in _TYPE_CASTS:
        target_dtype = _TYPE_CASTS[tensor.dtype]
        LOGGER.debug(f"Casting {tensor.dtype} tensor to {target_dtype.type}.")
        return tensor.astype(target_dtype.type)

    return tensor


def get_trt_profile_with_new_max_batch_size(
    trt_profile: TensorRTProfile, max_batch_size: int, batch_dim: int
) -> TensorRTProfile:
    """Create new TensorRT profile with maximum batch size.

    Args:
        trt_profile (TensorRTProfile): TensorRT Profile.
        max_batch_size (int): new maximum batch size.
        batch_dim (int): Batch dimension.

    Returns:
        Profile: New TensoRT Profile.
    """
    new_profile = TensorRTProfile()
    for input_name in trt_profile:
        max_shapes = list(trt_profile[input_name].max)
        max_shapes[batch_dim] = max_batch_size
        new_profile[input_name] = ShapeTuple(
            trt_profile[input_name].min, trt_profile[input_name].opt, tuple(max_shapes)
        )
    return new_profile
