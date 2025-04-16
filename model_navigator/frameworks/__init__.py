# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
"""Definition of Deep Learning frameworks related constants."""

from enum import Enum

from packaging.version import Version

try:
    import jax.experimental  # pytype: disable=import-error # noqa: F401

    _JAX_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _JAX_AVAILABLE = False

try:
    import torch  # pytype: disable=import-error # noqa: F401

    _TORCH_AVAILABLE = True
    _TORCH_VERSION = Version(torch.__version__)
except (ModuleNotFoundError, ImportError):
    _TORCH_AVAILABLE = False

try:
    import tensorflow  # pytype: disable=import-error

    _TF_VERSION = Version(tensorflow.__version__)

    if _TF_VERSION < Version("2.0.0"):
        _TF_AVAILABLE = False
    else:
        _TF_AVAILABLE = True
except (ModuleNotFoundError, AttributeError, ImportError):
    _TF_AVAILABLE = False


try:
    import tensorrt  # pytype: disable=import-error # noqa: F401

    _TRT_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _TRT_AVAILABLE = False


try:
    import modelopt  # pytype: disable=import-error # noqa: F401

    _MODEL_OPT_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _MODEL_OPT_AVAILABLE = False


class Framework(Enum):
    """Frameworks for models that are supported as input for `optimize` method."""

    TENSORFLOW = "tensorflow"
    TORCH = "torch"
    ONNX = "onnx"
    JAX = "jax"
    NONE = "none"
    TENSORRT = "tensorrt"


class Extension(Enum):
    """Definition of file extensions for produced serialized models."""

    ONNX = "onnx"
    PT = "pt"
    SAVEDMODEL = "savedmodel"
    TRT = "plan"


def is_torch_available() -> bool:
    """Check if torch is available.

    Returns:
        bool: True if torch is available.
    """
    return _TORCH_AVAILABLE


def is_torch2_available() -> bool:
    """Check if torch2 is available.

    Returns:
        bool: True if torch2 is available.
    """
    return _TORCH_AVAILABLE and _TORCH_VERSION >= Version("2.0.0")


def is_tf_available() -> bool:
    """Check if tensorflow is available.

    Returns:
        bool: True if tensorflow is available.
    """
    return _TF_AVAILABLE


def is_jax_available() -> bool:
    """Check if JAX is available.

    Returns:
        bool: True if JAX is available.
    """
    return _JAX_AVAILABLE


def is_trt_available() -> bool:
    """Check if TensorRT is available.

    Returns:
        bool: True if TensorRT is available.
    """
    return _TRT_AVAILABLE


def is_modelopt_available() -> bool:
    """Check if ModelOptimizer is available.

    Returns:
        bool: True if ModelOptimizer is available.
    """
    return _MODEL_OPT_AVAILABLE
