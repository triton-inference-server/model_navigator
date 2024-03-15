# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""Configuration of TensorRT platform supported models on Triton Inference Server."""

import dataclasses
from typing import Optional

from model_navigator.exceptions import ModelNavigatorWrongParameterError

from .base_model_config import BaseSpecializedModelConfig
from .common import Platform
from .internal import Backend


@dataclasses.dataclass
class TensorRTOptimization:
    """TensorRT possible optimizations.

    Args:
        cuda_graphs: Use CUDA graphs API to capture model operations and execute them more efficiently.
        gather_kernel_buffer_threshold: The backend may use a gather kernel to gather input data if the
                                        device has direct access to the source buffer and the destination
                                        buffer.
        eager_batching: Start preparing the next batch before the model instance is ready for the next inference.
    """

    cuda_graphs: bool = False
    gather_kernel_buffer_threshold: Optional[int] = None
    eager_batching: bool = False

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if not self.cuda_graphs and not self.gather_kernel_buffer_threshold and not self.eager_batching:
            raise ModelNavigatorWrongParameterError("At least one of the optimization options should be enabled.")


@dataclasses.dataclass()
class TensorRTModelConfig(BaseSpecializedModelConfig):
    """Specialized model config for TensorRT platform supported model.

    Args:
        platform: Override backend parameter with platform.
                  Possible options: Platform.TensorRTPlan
        optimization: Possible optimization for TensorRT models
    """

    platform: Optional[Platform] = None
    optimization: Optional[TensorRTOptimization] = None

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        super().__post_init__()
        if self.optimization and not isinstance(self.optimization, TensorRTOptimization):
            raise ModelNavigatorWrongParameterError("Unsupported optimization type provided.")

        if self.platform and self.platform != Platform.TensorRTPlan:
            raise ModelNavigatorWrongParameterError(f"Unsupported platform provided. Use: {Platform.TensorRTPlan}.")

    @property
    def backend(self) -> Backend:
        """Define backend value for config."""
        return Backend.TensorRT
