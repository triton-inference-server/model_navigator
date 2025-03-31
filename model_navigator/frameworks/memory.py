# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Memory management utilities for frameworks."""

from typing import Any

from model_navigator.frameworks import Framework, is_torch_available


def offload_model_to_cpu(model: Any, framework: Framework):
    """Offload model to CPU.

    Args:
        model: Model to offload.
        framework: Framework of model to offload.
    """
    if is_torch_available() and framework == Framework.TORCH:
        from model_navigator.frameworks.torch.utils import offload_torch_model_to_cpu

        offload_torch_model_to_cpu(model)
