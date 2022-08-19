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

import numpy as np

from model_navigator.model import ModelSignatureConfig

LOGGER = logging.getLogger(__name__)


def get_version():
    from polygraphy import mod

    trt = mod.lazy_import("tensorrt")
    trt_version = LooseVersion(trt.__version__)
    return trt_version


def rewrite_signature_config(signature: ModelSignatureConfig):
    """
    Rewrite the signature of model for TensorRT
    """
    rewritten_signature = ModelSignatureConfig(inputs={}, outputs={})
    if signature is None or signature.is_missing():
        return rewritten_signature

    for name, tensor in signature.inputs.items():
        rewritten_signature.inputs[name] = _cast_tensor(tensor)

    for name, tensor in signature.outputs.items():
        rewritten_signature.outputs[name] = _cast_tensor(tensor)

    return rewritten_signature


def _cast_tensor(tensor):
    """
    Cast TensorSpec object to supported type for TensorRT
    """
    trt_casts = {np.dtype(np.int64): np.int32}

    if tensor.dtype in trt_casts:
        LOGGER.debug(f"Casting {tensor.dtype} tensor to {trt_casts[tensor.dtype]}.")
        return tensor.astype(trt_casts[tensor.dtype])

    return tensor
