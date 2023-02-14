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
# noqa: D104
from model_navigator.commands.base import CommandStatus  # noqa: F401
from model_navigator.utils.framework import is_jax_available, is_tf_available, is_torch_available  # noqa: F401

from .config import (  # noqa: F401
    Format,
    JitType,
    MeasurementMode,
    OnnxConfig,
    ProfilerConfig,
    TensorFlowConfig,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TensorRTProfile,
    TorchConfig,
    TorchTensorRTConfig,
)

if is_torch_available():
    from . import torch  # noqa: F401

if is_tf_available():
    from . import tensorflow  # noqa: F401

if is_tf_available() and is_jax_available():
    from . import jax  # noqa: F401

from . import onnx  # noqa: F401
from . import package  # noqa: F401
from . import triton  # noqa: F401
