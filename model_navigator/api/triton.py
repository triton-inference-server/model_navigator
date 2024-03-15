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
"""Public API definition for Triton related functionality."""

from model_navigator.triton import model_repository  # noqa: F401
from model_navigator.triton.specialized_configs import (  # noqa: F401
    AutoMixedPrecisionAccelerator,
    BaseSpecializedModelConfig,
    DeviceKind,
    DynamicBatcher,
    GPUIOAccelerator,
    InputTensorFormat,
    InputTensorSpec,
    InstanceGroup,
    ModelWarmup,
    ModelWarmupInput,
    ModelWarmupInputDataType,
    ONNXModelConfig,
    ONNXOptimization,
    OpenVINOAccelerator,
    OutputTensorSpec,
    Platform,
    PythonModelConfig,
    PyTorchModelConfig,
    QueuePolicy,
    SequenceBatcher,
    SequenceBatcherControl,
    SequenceBatcherControlInput,
    SequenceBatcherControlKind,
    SequenceBatcherInitialState,
    SequenceBatcherState,
    SequenceBatcherStrategyDirect,
    SequenceBatcherStrategyOldest,
    TensorFlowModelConfig,
    TensorFlowOptimization,
    TensorRTAccelerator,
    TensorRTModelConfig,
    TensorRTOptimization,
    TensorRTOptPrecision,
    TimeoutAction,
)
