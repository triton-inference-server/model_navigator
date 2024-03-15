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
"""Configuration of ONNX models supported on Triton Inference Server."""

import dataclasses
from typing import Optional, Union

from model_navigator.exceptions import ModelNavigatorWrongParameterError

from .base_model_config import BaseSpecializedModelConfig
from .common import Platform, TensorRTAccelerator
from .internal import Backend


@dataclasses.dataclass
class OpenVINOAccelerator:
    """OpenVINO optimization.

    Currently empty - no arguments required.
    """

    pass


@dataclasses.dataclass
class ONNXOptimization:
    """ONNX possible optimizations.

    Args:
        accelerator: Execution accelerator for model
    """

    accelerator: Union[OpenVINOAccelerator, TensorRTAccelerator]

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.accelerator and type(self.accelerator) not in [OpenVINOAccelerator, TensorRTAccelerator]:
            raise ModelNavigatorWrongParameterError("Unsupported accelerator type provided.")


@dataclasses.dataclass
class ONNXModelConfig(BaseSpecializedModelConfig):
    """Specialized model config for ONNX backend supported model.

    Args:
        platform: Override backend parameter with platform.
                  Possible options: Platform.ONNXRuntimeONNX
        optimization: Possible optimization for ONNX models
    """

    platform: Optional[Platform] = None
    optimization: Optional[ONNXOptimization] = None

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        super().__post_init__()
        if self.optimization and not isinstance(self.optimization, ONNXOptimization):
            raise ModelNavigatorWrongParameterError("Unsupported optimization type provided.")

        if self.platform and self.platform != Platform.ONNXRuntimeONNX:
            raise ModelNavigatorWrongParameterError(f"Unsupported platform provided. Use: {Platform.ONNXRuntimeONNX}.")

    @property
    def backend(self) -> Backend:
        """Define backend value for config."""
        return Backend.ONNXRuntime
