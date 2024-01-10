# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Validation utilities for device."""

import re
from typing import Optional

from model_navigator.exceptions import ModelNavigatorConfigurationError


def validate_device_string(device: str):
    """Validate device torch-like string.

    Args:
        device: Device string e.g. cuda:0, cuda:1, cpu
    """
    pattern = r"^(cuda:\d+|cuda|cpu)$"
    if not bool(re.match(pattern, device)):
        raise ModelNavigatorConfigurationError("device must be 'cpu' or in format 'cuda:<device_id>'")


def validate_device_string_for_cuda(device: str):
    """Validate device torch-like string for cuda.

    Args:
        device: Device string e.g. cuda:0, cuda:1, cpu
    """
    pattern = r"^(cuda:\d+|cuda)$"
    if not bool(re.match(pattern, device)):
        raise ModelNavigatorConfigurationError("device must be 'cuda' or in format 'cuda:<device_id>'")


def get_id_from_device_string(device: str) -> Optional[int]:
    """Get device id from device string.

    Args:
        device: Device string e.g. cuda:0, cuda:1, cpu

    Returns:
        Device id or None if not found.
    """
    pattern = r"^cuda:(\d+)$"
    match = re.match(pattern, device)
    if match:
        return int(match.group(1))
    return None
