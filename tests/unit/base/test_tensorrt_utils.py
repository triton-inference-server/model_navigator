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
import numpy as np

from model_navigator.api.config import ShapeTuple, TensorRTProfile
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.utils.common import optimal_batch_size


def test_cast_type_return_current_type_when_has_no_cast(mocker):
    mocker.patch("model_navigator.frameworks.tensorrt.utils.get_version", return_value="8.6")
    dtypes = [np.dtype("int32"), np.dtype("float32"), np.dtype("object")]
    for dtype in dtypes:
        assert dtype == tensorrt_utils.cast_type(dtype)


def test_cast_type_return_new_type_when_has_cast_before_8_6(mocker):
    mocker.patch("model_navigator.frameworks.tensorrt.utils.get_version", return_value="8.6")
    assert np.dtype("int32") == tensorrt_utils.cast_type(np.dtype("int64"))
    assert np.dtype("float32") == tensorrt_utils.cast_type(np.dtype("float64"))
    assert np.dtype("uint32") == tensorrt_utils.cast_type(np.dtype("uint64"))


def test_cast_type_return_new_type_when_has_cast_after_8_6(mocker):
    mocker.patch("model_navigator.frameworks.tensorrt.utils.get_version", return_value="9.0")
    assert np.dtype("int64") == tensorrt_utils.cast_type(np.dtype("int64"))
    assert np.dtype("float32") == tensorrt_utils.cast_type(np.dtype("float64"))
    assert np.dtype("uint32") == tensorrt_utils.cast_type(np.dtype("uint64"))


def test_cast_tensor_is_not_changed_when_tensor_has_no_cast_type(mocker):
    mocker.patch("model_navigator.frameworks.tensorrt.utils.get_version", return_value="8.6")
    tensor = np.zeros(shape=(1,), dtype=np.dtype("int32"))
    modified_tensor = tensorrt_utils.cast_tensor(tensor)

    assert modified_tensor.dtype == tensor.dtype
    assert modified_tensor.shape == tensor.shape


def test_cast_tensor_is_changed_when_tensor_cast_type_before_8_6(mocker):
    mocker.patch("model_navigator.frameworks.tensorrt.utils.get_version", return_value="8.6")
    test_cases = [
        ("int64", "int32"),
        ("uint64", "uint32"),
        ("float64", "float32"),
    ]
    for input_type, expected_type in test_cases:
        tensor = np.zeros(shape=(1,), dtype=np.dtype(input_type))
        modified_tensor = tensorrt_utils.cast_tensor(tensor)

        assert modified_tensor.dtype == np.dtype(expected_type)
        assert modified_tensor.shape == tensor.shape


def test_cast_tensor_is_changed_when_tensor_cast_type_after_8_6(mocker):
    mocker.patch("model_navigator.frameworks.tensorrt.utils.get_version", return_value="9.0")
    test_cases = [
        ("int64", "int64"),
        ("uint64", "uint32"),
        ("float64", "float32"),
    ]
    for input_type, expected_type in test_cases:
        tensor = np.zeros(shape=(1,), dtype=np.dtype(input_type))
        modified_tensor = tensorrt_utils.cast_tensor(tensor)

        assert modified_tensor.dtype == np.dtype(expected_type)
        assert modified_tensor.shape == tensor.shape


def test_get_trt_profile_return_updates_batch_size_when_max_bs_equal_to_1():
    batch_dim = 0
    ref_profile = TensorRTProfile({
        "input__0": ShapeTuple((1, 2), (1, 3), (1, 4)),
        "input__1": ShapeTuple((1, 3), (1, 4), (1, 5)),
    })
    old_profile = TensorRTProfile({
        "input__0": ShapeTuple((1, 2), (1, 3), (1, 4)),
        "input__1": ShapeTuple((1, 3), (1, 4), (1, 5)),
    })
    updated_profile = tensorrt_utils.get_trt_profile_with_new_max_batch_size(old_profile, 1, batch_dim)
    for old_shape, updated_shape, ref_shape in zip(
        old_profile.values(), updated_profile.values(), ref_profile.values()
    ):
        assert old_shape.min == updated_shape.min == ref_shape.min
        assert old_shape.opt == updated_shape.opt == ref_shape.opt
        assert old_shape.max == updated_shape.max == ref_shape.max


def test_get_trt_profile_return_updates_batch_size_when_max_bs_greater_than_1():
    batch_dim = 0
    old_max_bs, new_max_bs = 16, 32
    ref_profile = TensorRTProfile({
        "input__0": ShapeTuple((1, 2), (2, 3), (old_max_bs, 4)),
        "input__1": ShapeTuple((1, 3), (1, 4), (old_max_bs, 5)),
    })
    old_profile = TensorRTProfile({
        "input__0": ShapeTuple((1, 2), (2, 3), (old_max_bs, 4)),
        "input__1": ShapeTuple((1, 3), (1, 4), (old_max_bs, 5)),
    })
    updated_profile = tensorrt_utils.get_trt_profile_with_new_max_batch_size(old_profile, new_max_bs, batch_dim)
    for old_shape, updated_shape, ref_shape in zip(
        old_profile.values(), updated_profile.values(), ref_profile.values()
    ):
        assert old_shape.min == updated_shape.min == ref_shape.min

        for i in range(len(old_shape.opt)):
            if i == batch_dim:
                assert old_shape.opt[i] == ref_shape.opt[i]
                assert updated_shape.opt[i] == optimal_batch_size(new_max_bs)
            else:
                assert old_shape.opt[i] == updated_shape.opt[i] == ref_shape.opt[i]

        for i in range(len(old_shape.max)):
            if i == batch_dim:
                assert old_shape.max[i] == ref_shape.max[i] == old_max_bs
                assert updated_shape.max[i] == new_max_bs
            else:
                assert old_shape.max[i] == updated_shape.max[i] == ref_shape.max[i]
