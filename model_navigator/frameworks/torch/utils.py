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
"""Torch utils."""

import gc
from typing import Optional

from model_navigator.core.logger import LOGGER
from model_navigator.utils.module import lazy_import

torch = lazy_import("torch")


def get_module_device(module: "torch.nn.Module") -> Optional["torch.device"]:
    """Get the device of the given module.

    Args:
        module: Module to get the device of.

    Returns:
        The device of module based on parameters. If not parameters, returns None.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def offload_torch_model_to_cpu(model: "torch.nn.Module"):
    """Offload PyTorch model to CPU.

    Args:
        model: PyTorch model to offload.
    """
    LOGGER.info("Offloading model to CPU")
    model.to("cpu")

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
