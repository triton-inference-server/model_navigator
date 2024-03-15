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

from model_navigator.frameworks import is_jax_available, is_tf_available, is_torch2_available, is_torch_available

if is_torch_available():
    from . import (
        torch2onnx,  # noqa: F401
        torch2torchscript,  # noqa: F401
    )

if is_torch2_available():
    from . import (
        torch2dynamo_onnx,  # noqa: F401
        torch2exportedprogram,  # noqa: F401
    )

if is_tf_available():
    from . import (
        keras2savedmodel,  # noqa: F401
        savedmodel2savedmodel,  # noqa: F401
    )

if is_tf_available() and is_jax_available():
    from . import jax2savedmodel  # noqa: F401
