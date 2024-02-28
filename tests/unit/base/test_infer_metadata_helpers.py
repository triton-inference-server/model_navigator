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

import numpy
import pytest

from model_navigator.api.config import TensorRTProfile, TensorType
from model_navigator.commands.infer_metadata import (
    _assert_all_inputs_have_same_pytree_metadata,
    _extract_max_batch_size,
    _get_metadata_from_axes_shapes,
    _get_trt_profile_from_axes_shapes,
)
from model_navigator.core.tensor import PyTreeMetadata, TensorSpec
from model_navigator.exceptions import ModelNavigatorUserInputError


def test_extract_max_batch_size_return_correct_value_when_multiple_values_passed():
    input_name = "input_0"
    max_batch_size = 999
    batch_dim = 0
    axes_shapes = {
        input_name: {
            0: [5, max_batch_size, 1, 3, 7],  # batch dimension
            1: [224, 224, 224, 224, 224],
            2: [224, 224, 224, 224, 224],
            3: [3, 3, 3, 3, 3],
        }
    }

    batch_size = _extract_max_batch_size(axes_shapes=axes_shapes, batch_dim=batch_dim)

    assert batch_size == max_batch_size


def test_get_trt_profile_return_correct_shapes_when_axes_shapes_passed():
    input_names = ["input_0", "input_1"]
    axes_shapes = {
        input_names[0]: {0: [1, 2, 2], 1: [1, 2, 3]},
        input_names[1]: {0: [1, 2, 3], 1: [224, 356, 448], 2: [224, 356, 448], 3: [3, 3, 3]},
    }
    batch_dim = 0

    expected_trt_profile = (
        TensorRTProfile()
        .add(input_names[0], min=(1, 1), opt=(2, 2), max=(2, 3))
        .add(input_names[1], min=(1, 224, 224, 3), opt=(2, 356, 356, 3), max=(3, 448, 448, 3))
    )

    trt_profile = _get_trt_profile_from_axes_shapes(axes_shapes=axes_shapes, batch_dim=batch_dim)

    for input_name in input_names:
        assert input_name in trt_profile
        assert trt_profile[input_name].min == expected_trt_profile[input_name].min
        assert trt_profile[input_name].opt == expected_trt_profile[input_name].opt
        assert trt_profile[input_name].max == expected_trt_profile[input_name].max


def test_get_trt_profile_return_correct_shapes_when_axes_shapes_and_config_max_batch_size_passed():
    input_names = ["input_0", "input_1"]
    axes_shapes = {
        input_names[0]: {0: [1, 2, 2], 1: [1, 2, 3]},
        input_names[1]: {0: [1, 2, 3], 1: [224, 356, 448], 2: [224, 356, 448], 3: [3, 3, 3]},
    }
    batch_dim = 0
    config_max_batch_size = 5

    expected_trt_profile = (
        TensorRTProfile()
        .add(input_names[0], min=(1, 1), opt=(2, 2), max=(config_max_batch_size, 3))
        .add(input_names[1], min=(1, 224, 224, 3), opt=(2, 356, 356, 3), max=(config_max_batch_size, 448, 448, 3))
    )

    trt_profile = _get_trt_profile_from_axes_shapes(
        axes_shapes=axes_shapes, batch_dim=batch_dim, config_max_batch_size=config_max_batch_size
    )

    for input_name in input_names:
        assert input_name in trt_profile
        assert trt_profile[input_name].min == expected_trt_profile[input_name].min
        assert trt_profile[input_name].opt == expected_trt_profile[input_name].opt
        assert trt_profile[input_name].max == expected_trt_profile[input_name].max


def test_get_metadata_return_correct_data_from_axes_shapes_when_with_valid_shapes_passed():
    batch_dim = 0
    input_name = "input_0"
    dtype_name = "float64"
    dtypes = {input_name: dtype_name}
    axes_shapes = {
        input_name: {0: [1, 1, 1, 1, 1], 1: [224, 224, 224, 224, 224], 2: [224, 224, 224, 224, 224], 3: [3, 3, 3, 3, 3]}
    }

    expected_metadata = {
        input_name: TensorSpec(name=input_name, shape=(-1, 224, 224, 3), dtype=numpy.dtype(dtype_name), optional=False)
    }
    metadata = _get_metadata_from_axes_shapes(
        pytree_metadata=PyTreeMetadata(None, TensorType.NUMPY),
        axes_shapes=axes_shapes,
        batch_dim=batch_dim,
        dtypes=dtypes,
    )

    assert metadata == expected_metadata


def test_assert_all_inputs_have_same_pytree_metadata_raise_no_exception_when_inputs_have_same_metadata():
    valid_dataloaders = [
        [
            numpy.zeros(2),
            numpy.zeros(2),
        ],
        [
            {"input_0": numpy.zeros(2), "input_1": numpy.zeros(3)},
            {"input_1": numpy.zeros(3), "input_0": numpy.zeros(2)},
        ],
        [
            (numpy.zeros(2), True),
            (numpy.zeros(2), True),
        ],
    ]

    for dataloader in valid_dataloaders:
        pytree_metadata = PyTreeMetadata.from_sample(dataloader[0], TensorType.NUMPY, prefix="dummy")

        _assert_all_inputs_have_same_pytree_metadata(dataloader, pytree_metadata)


def test_assert_all_inputs_have_same_pytree_metadata_raise_exception_when_inputs_have_different_metadata():
    invalid_dataloaders = [
        [
            numpy.zeros(2),
            (numpy.zeros(2),),
        ],
        [
            {"input_0": numpy.zeros(2), "input_1": numpy.zeros(3)},
            {"input_0": numpy.zeros(2), "input_1": numpy.zeros(3), "input_2": numpy.zeros(3)},
        ],
        [
            (numpy.zeros(2), True),
            (numpy.zeros(2), False),
        ],
        [
            (numpy.zeros(2), False, 1.0),
            (numpy.zeros(2), False, 2.0),
        ],
    ]

    for dataloader in invalid_dataloaders:
        pytree_metadata = PyTreeMetadata.from_sample(dataloader[0], TensorType.NUMPY, prefix="dummy")

        with pytest.raises(ModelNavigatorUserInputError):
            _assert_all_inputs_have_same_pytree_metadata(dataloader, pytree_metadata)
