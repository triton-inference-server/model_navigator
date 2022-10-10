# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from model_navigator import TensorSpec
from model_navigator.utils import tensorrt


def test_cast_type_return_current_type_when_has_no_cast():
    dtypes = [np.dtype("int32"), np.dtype("float32"), np.dtype("object")]
    for dtype in dtypes:
        assert dtype == tensorrt.cast_type(dtype)


def test_cast_type_return_new_type_when_has_cast():
    assert np.dtype("int32") == tensorrt.cast_type(np.dtype("int64"))


def test_cast_tensor_is_not_changed_when_tensor_has_no_cast_type():
    tensor = TensorSpec(name="Tensor", shape=(-1,), dtype=np.dtype("int32"))
    modified_tensor = tensorrt.cast_tensor(tensor)

    assert modified_tensor.dtype == tensor.dtype
    assert modified_tensor.shape == tensor.shape
    assert modified_tensor.name == tensor.name


def test_cast_tensor_is_changed_when_tensor_cast_type():
    tensor = TensorSpec(name="Tensor", shape=(-1,), dtype=np.dtype("int64"))
    modified_tensor = tensorrt.cast_tensor(tensor)

    assert modified_tensor.dtype == np.dtype("int32")
    assert modified_tensor.shape == tensor.shape
    assert modified_tensor.name == tensor.name
