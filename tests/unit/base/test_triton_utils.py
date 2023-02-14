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
import numpy as np

from model_navigator.triton.utils import input_tensor_from_metadata, output_tensor_from_metadata
from model_navigator.utils.tensor import TensorMetadata


def test_input_tensor_from_metadata_return_input_tensor_when_no_batching():
    metadata = TensorMetadata()

    metadata.add(name="input_1", shape=(224, 224, 3), dtype=np.float32)
    metadata.add(name="input_2", shape=(-1, -1), dtype=np.int32)

    tensors = input_tensor_from_metadata(input_metadata=metadata, batching=False)

    assert len(tensors) == 2

    assert tensors[0].name == "input_1"
    assert tensors[0].shape == (224, 224, 3)
    assert tensors[0].dtype == np.dtype(np.float32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].optional is False
    assert tensors[0].format is None
    assert tensors[0].allow_ragged_batch is False

    assert tensors[1].name == "input_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.int32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].optional is False
    assert tensors[1].format is None
    assert tensors[1].allow_ragged_batch is False


def test_input_tensor_from_metadata_return_input_tensor_when_batching():
    metadata = TensorMetadata()

    metadata.add(name="input_1", shape=(-1, 224, 224, 3), dtype=np.float32)
    metadata.add(name="input_2", shape=(-1, -1, -1), dtype=np.int32)

    tensors = input_tensor_from_metadata(input_metadata=metadata, batching=True)

    assert len(tensors) == 2

    assert tensors[0].name == "input_1"
    assert tensors[0].shape == (224, 224, 3)
    assert tensors[0].dtype == np.dtype(np.float32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].optional is False
    assert tensors[0].format is None
    assert tensors[0].allow_ragged_batch is False

    assert tensors[1].name == "input_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.int32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].optional is False
    assert tensors[1].format is None
    assert tensors[1].allow_ragged_batch is False


def test_output_tensor_from_metadata_return_input_tensor_when_no_batching():
    metadata = TensorMetadata()

    metadata.add(name="output_1", shape=(1000,), dtype=np.int32)
    metadata.add(name="output_2", shape=(-1, -1), dtype=np.float32)

    tensors = output_tensor_from_metadata(output_metadata=metadata, batching=False)

    assert len(tensors) == 2

    assert tensors[0].name == "output_1"
    assert tensors[0].shape == (1000,)
    assert tensors[0].dtype == np.dtype(np.int32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].label_filename is None

    assert tensors[1].name == "output_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.float32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].label_filename is None


def test_output_tensor_from_metadata_return_input_tensor_when_batching():
    metadata = TensorMetadata()

    metadata.add(name="output_1", shape=(-1, 1000), dtype=np.int32)
    metadata.add(name="output_2", shape=(-1, -1, -1), dtype=np.float32)

    tensors = output_tensor_from_metadata(output_metadata=metadata, batching=True)

    assert len(tensors) == 2

    assert tensors[0].name == "output_1"
    assert tensors[0].shape == (1000,)
    assert tensors[0].dtype == np.dtype(np.int32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].label_filename is None

    assert tensors[1].name == "output_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.float32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].label_filename is None
