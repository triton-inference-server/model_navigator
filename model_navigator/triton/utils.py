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
"""Utilities related with Triton Inference Server."""
from typing import List

from model_navigator.utils.tensor import TensorMetadata

from .specialized_configs import InputTensorSpec, OutputTensorSpec


def input_tensor_from_metadata(input_metadata: TensorMetadata, batching: bool = True) -> List:
    """Generate list of input tensors based on TensorMetadata.

    Args:
        input_metadata: Model inputs metadata
        batching: Flag indicating if input metadata contain batch in shape

    Returns:
        List of input tensors
    """
    input_tensors = []
    for metadata in input_metadata.values():
        shape = metadata.shape[1:] if batching else metadata.shape
        tensor = InputTensorSpec(name=metadata.name, dtype=metadata.dtype, shape=shape)
        input_tensors.append(tensor)
    return input_tensors


def output_tensor_from_metadata(output_metadata: TensorMetadata, batching: bool = True) -> List:
    """Generate list of output tensors based on TensorMetadata.

    Args:
        output_metadata: Model outputs metadata
        batching: Flag indicating if output metadata contain batch in shape

    Returns:
        List of output tensors
    """
    output_tensors = []
    for metadata in output_metadata.values():
        shape = metadata.shape[1:] if batching else metadata.shape
        tensor = OutputTensorSpec(name=metadata.name, dtype=metadata.dtype, shape=shape)
        output_tensors.append(tensor)
    return output_tensors
