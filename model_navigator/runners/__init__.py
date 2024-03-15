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

from model_navigator.frameworks import (
    is_jax_available,
    is_tf_available,
    is_torch2_available,
    is_torch_available,
    is_trt_available,
)

from .onnx import register_onnx_runners
from .python import register_python_runners
from .registry import load_runners_from_entry_points, register_runner  # noqa: F401

register_python_runners()
register_onnx_runners()

if is_trt_available():
    from .tensorrt import register_tensorrt_runners

    register_tensorrt_runners()

if is_torch_available():
    from .torch import register_torch_runners

    register_torch_runners()

if is_torch2_available():
    from .torch import register_torch2_runners

    register_torch2_runners()

if is_tf_available():
    from .tensorflow import register_tensorflow_runners

    register_tensorflow_runners()

if is_tf_available() and is_jax_available():
    from .jax import register_jax_runners

    register_jax_runners()

load_runners_from_entry_points()
