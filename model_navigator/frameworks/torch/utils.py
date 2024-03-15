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
"""Torch utils."""

from typing import Optional, Sequence, Tuple

from model_navigator.api.config import CustomConfig, Format, TorchTensorRTConfig
from model_navigator.core.logger import LOGGER


def update_allowed_batching_parameters(
    target_formats: Tuple[Format, ...], custom_configs: Optional[Sequence[CustomConfig]]
):
    """Update target formats and custom configs to disable Torch-TensorRT when batching is True."""
    target_formats = tuple(tf for tf in target_formats if tf != Format.TORCH_TRT)
    if custom_configs:
        custom_configs = [config for config in custom_configs if not isinstance(config, TorchTensorRTConfig)]

    LOGGER.warning(
        "When `batching` is True, Torch-TensorRT target format is disabled in the default configuration.\n"
        "Explicitly set `nav.Format.TORCH_TRT` in `target_formats` to use it."
    )

    return target_formats, custom_configs
