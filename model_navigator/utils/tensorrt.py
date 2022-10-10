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
from distutils.version import LooseVersion
from typing import Tuple

import numpy as np

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.model import ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.utils import enums

LOGGER = logging.getLogger(__name__)
_TYPE_CASTS = {np.dtype(np.int64): np.dtype(np.int32)}


def get_version():
    from polygraphy import mod

    trt = mod.lazy_import("tensorrt")
    trt_version = LooseVersion(trt.__version__)
    return trt_version


def filter_deprecated_precision(target_precisions: Tuple[TensorRTPrecision, ...]) -> Tuple:
    filtered_precision = set()
    target_precisions = enums.parse(target_precisions, TensorRTPrecision)
    for precision in target_precisions:
        mapped_precision = TensorRTPrecision(precision)
        if mapped_precision == TensorRTPrecision.TF32:
            LOGGER.warning(f"The {mapped_precision} is deprecated. Using {TensorRTPrecision.FP32.value}.")
            filtered_precision.add(TensorRTPrecision.FP32)
        else:
            filtered_precision.add(mapped_precision)

    return tuple(filtered_precision)


def rewrite_signature_config(signature: ModelSignatureConfig):
    """
    Rewrite the signature of model for TensorRT
    """
    rewritten_signature = ModelSignatureConfig(inputs={}, outputs={})
    if signature is None or signature.is_missing():
        return rewritten_signature

    for name, tensor in signature.inputs.items():
        rewritten_signature.inputs[name] = cast_tensor(tensor)

    for name, tensor in signature.outputs.items():
        rewritten_signature.outputs[name] = cast_tensor(tensor)

    return rewritten_signature


def cast_type(dtype: np.dtype) -> np.dtype:
    """
    Cast type and return new dtype
    """
    if dtype in _TYPE_CASTS:
        return _TYPE_CASTS[dtype]

    return dtype


def cast_tensor(tensor: TensorSpec) -> TensorSpec:
    """
    Cast TensorSpec object to supported type for TensorRT
    """
    if tensor.dtype in _TYPE_CASTS:
        target_dtype = _TYPE_CASTS[tensor.dtype]
        LOGGER.debug(f"Casting {tensor.dtype} tensor to {target_dtype.type}.")
        return tensor.astype(target_dtype.type)

    return tensor
