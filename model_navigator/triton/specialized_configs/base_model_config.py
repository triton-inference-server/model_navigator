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
"""Configuration of base model config on Triton Inference Server."""

import abc
import dataclasses
from typing import Dict, List, Optional, Sequence, Union

from model_navigator.exceptions import ModelNavigatorWrongParameterError

from .common import DynamicBatcher, InputTensorSpec, InstanceGroup, ModelWarmup, OutputTensorSpec, SequenceBatcher
from .internal import Backend


@dataclasses.dataclass
class BaseSpecializedModelConfig(abc.ABC):
    """Common fields for specialized model configs.

    Read more in Triton Inference server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto

    Args:
        max_batch_size: The maximal batch size that would be handled by model.
        batching: Flag to enable/disable batching for model.
        default_model_filename: Optional filename of the model file to use.
        batcher: Configuration of Dynamic Batching for the model.
        instance_groups: Instance groups configuration for multiple instances of the model
        parameters: Custom parameters for model or backend
        response_cache: Flag to enable/disable response cache for the model
        warmup: Warmup configuration for model
    """

    max_batch_size: int = 4
    batching: bool = True
    default_model_filename: Optional[str] = None
    batcher: Union[DynamicBatcher, SequenceBatcher] = dataclasses.field(default_factory=DynamicBatcher)
    instance_groups: List[InstanceGroup] = dataclasses.field(default_factory=lambda: [])
    parameters: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    response_cache: bool = False
    warmup: Dict[str, ModelWarmup] = dataclasses.field(default_factory=lambda: {})
    inputs: Sequence[InputTensorSpec] = dataclasses.field(default_factory=lambda: [])
    outputs: Sequence[OutputTensorSpec] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        """Validate the configuration for early error handling."""
        if self.batching and self.max_batch_size <= 0:
            raise ModelNavigatorWrongParameterError("The `max_batch_size` must be greater or equal to 1.")

        if type(self.batcher) not in [DynamicBatcher, SequenceBatcher]:
            raise ModelNavigatorWrongParameterError("Unsupported batcher type provided.")

        if self.backend != Backend.TensorRT and any(group.profile for group in self.instance_groups):
            raise ModelNavigatorWrongParameterError(
                "Invalid `profile` option. The value can be set only for `backend=Backend.TensorRT`"
            )

    @property
    @abc.abstractmethod
    def backend(self) -> Backend:
        """Backend property that has to be overridden by specialized configs."""
        raise NotImplementedError()
