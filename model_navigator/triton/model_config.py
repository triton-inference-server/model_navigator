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
"""ModelConfig related objects."""

import dataclasses
from typing import Dict, List, Optional, Sequence, Union

from model_navigator.exceptions import ModelNavigatorWrongParameterError

from .specialized_configs import (
    Backend,
    DynamicBatcher,
    InputTensorSpec,
    InstanceGroup,
    ModelWarmup,
    ONNXOptimization,
    OutputTensorSpec,
    Platform,
    SequenceBatcher,
    TensorFlowOptimization,
    TensorRTOptimization,
)


@dataclasses.dataclass
class ModelConfig:
    """Triton Model Config dataclass for simplification and specialization of protobuf config generation.

    Read more in Triton Inference server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
    """

    model_name: str
    model_version: int = 1
    backend: Optional[Backend] = None
    platform: Optional[Platform] = None
    default_model_filename: Optional[str] = None
    max_batch_size: int = 4
    batching: bool = True
    batcher: Union[DynamicBatcher, SequenceBatcher] = dataclasses.field(default_factory=DynamicBatcher)
    instance_groups: List[InstanceGroup] = dataclasses.field(default_factory=lambda: [])
    parameters: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    inputs: Optional[Sequence[InputTensorSpec]] = None
    outputs: Optional[Sequence[OutputTensorSpec]] = None
    optimization: Optional[Union[TensorRTOptimization, TensorFlowOptimization, ONNXOptimization]] = None
    response_cache: Optional[bool] = None
    decoupled: Optional[bool] = None
    warmup: Dict[str, ModelWarmup] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate the configuration for early error handling."""
        if not self.backend and not self.platform:
            raise ModelNavigatorWrongParameterError("Backend or platform has to be defined. None was provided.")

        if self.batching and self.max_batch_size <= 0:
            raise ModelNavigatorWrongParameterError("The `max_batch_size` must be greater or equal to 1.")

        if self.backend != Backend.TensorRT and any(group.profile for group in self.instance_groups):
            raise ModelNavigatorWrongParameterError(
                "Invalid `profile` option. The value can be set only for `backend=Backend.TensorRT`"
            )

        if type(self.batcher) not in [DynamicBatcher, SequenceBatcher]:
            raise ModelNavigatorWrongParameterError("Unsupported batcher type provided.")

        if self.optimization and type(self.optimization) not in [
            TensorRTOptimization,
            TensorFlowOptimization,
            ONNXOptimization,
        ]:
            raise ModelNavigatorWrongParameterError("Unsupported optimization type provided.")

        if self.inputs and self.warmup:
            for warmup in self.warmup.values():
                if len(self.inputs) != len(warmup.inputs):
                    raise ModelNavigatorWrongParameterError("Length of warmup inputs not equal defined inputs.")

                missing_inputs = []
                for inpt in self.inputs:
                    warmup_input = warmup.inputs.get(inpt.name)
                    if warmup_input is None:
                        missing_inputs.append(inpt.name)
                    elif warmup_input.dtype != inpt.dtype:
                        raise ModelNavigatorWrongParameterError(
                            f"Incompatible data types for {inpt.name}. Expected: {inpt.dtype}. Got: {warmup_input.dtype}."
                        )

                if missing_inputs:
                    raise ModelNavigatorWrongParameterError(
                        f"Missing defined warmup inputs: {', '.join(missing_inputs)}."
                    )
