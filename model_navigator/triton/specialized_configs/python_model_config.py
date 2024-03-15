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
"""Configuration of Python backend supported models on Triton Inference Server."""

import dataclasses
from typing import Sequence

from .base_model_config import BaseSpecializedModelConfig
from .common import InputTensorSpec, OutputTensorSpec
from .internal import Backend


@dataclasses.dataclass()
class PythonModelConfig(BaseSpecializedModelConfig):
    """Specialized model config for Python backend supported model.

    Args:
        inputs: Required definition of model inputs
        outputs: Required definition of model outputs
    """

    inputs: Sequence[InputTensorSpec] = dataclasses.field(default_factory=lambda: [])
    outputs: Sequence[OutputTensorSpec] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        """Validate the configuration for early error handling."""
        super().__post_init__()
        assert len(self.inputs) > 0, "Model inputs definition is required for Python backend."
        assert len(self.outputs) > 0, "Model outputs definition is required for Python backend."

    @property
    def backend(self) -> Backend:
        """Define backend value for config."""
        return Backend.Python
