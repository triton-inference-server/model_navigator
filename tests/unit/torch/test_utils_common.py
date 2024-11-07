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


import pytest
import torch  # pytype: disable=import-error

from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.utils.common import str_to_torch_dtype


def test_str_to_torch_dtype_float32():
    assert str_to_torch_dtype("torch.float32") == torch.float32


def test_str_to_torch_dtype_float64():
    assert str_to_torch_dtype("torch.float64") == torch.float64


def test_str_to_torch_dtype_int32():
    assert str_to_torch_dtype("torch.int32") == torch.int32


def test_str_to_torch_dtype_int64():
    assert str_to_torch_dtype("torch.int64") == torch.int64


def test_str_to_torch_dtype_uint8():
    assert str_to_torch_dtype("torch.uint8") == torch.uint8


def test_str_to_torch_dtype_bool():
    assert str_to_torch_dtype("torch.bool") == torch.bool


def test_str_to_torch_dtype_invalid():
    with pytest.raises(ModelNavigatorUserInputError):
        str_to_torch_dtype("invalid_dtype")


def test_str_to_torch_dtype_invalid_prefix():
    with pytest.raises(ModelNavigatorUserInputError):
        str_to_torch_dtype("pytorch.float32")


def test_str_to_torch_dtype_invalid_suffix():
    with pytest.raises(ModelNavigatorUserInputError):
        str_to_torch_dtype("torch.invalid_suffix")


def test_str_to_torch_dtype_empty():
    with pytest.raises(ModelNavigatorUserInputError):
        str_to_torch_dtype("")
