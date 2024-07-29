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
"""Multiprofile TensorRT runner tests"""

from unittest.mock import MagicMock, patch

from model_navigator.runners.tensorrt import TensorRTRunner


def test_find_best_profile_for_single_input_return_idx_1():
    shapes = {
        0: {"input__0": [(1, 5), (100000, 5), (200000, 5)]},
        1: {"input__0": [(1, 5), (1, 5), (1, 5)]},
        2: {"input__0": [(200, 5), (200, 5), (250, 5)]},
        3: {"input__0": [(300, 5), (300, 5), (350, 5)]},
    }

    input_tensor = {"input__0": MagicMock(shape=[1, 5])}

    with patch.multiple(
        TensorRTRunner,
        __init__=MagicMock(return_value=None),
        __del__=MagicMock(return_value=None),
        get_num_optimization_profiles=lambda x: 4,
        get_input_metadata=lambda s: [1],
        is_binding_input=lambda s, name: True,
        get_binding_name=lambda s, idx: "input__0",
        get_profile_shape=lambda s, idx, name: shapes[idx][name],
    ):
        runner = TensorRTRunner()

        profile = runner.find_best_profile(input_tensor)
        assert profile == 1


def test_find_best_profile_for_single_input_return_idx_2():
    shapes = {
        0: {"input__0": [(1, 5), (100000, 5), (200000, 5)]},
        1: {"input__0": [(200, 5), (200, 5), (250, 5)]},
        2: {"input__0": [(1, 5), (1, 5), (1, 5)]},
        3: {"input__0": [(300, 5), (300, 5), (350, 5)]},
    }

    input_tensor = {"input__0": MagicMock(shape=[1, 5])}

    with patch.multiple(
        TensorRTRunner,
        __init__=MagicMock(return_value=None),
        __del__=MagicMock(return_value=None),
        get_num_optimization_profiles=lambda x: 4,
        get_input_metadata=lambda s: [1],
        is_binding_input=lambda s, name: True,
        get_binding_name=lambda s, idx: "input__0",
        get_profile_shape=lambda s, idx, name: shapes[idx][name],
    ):
        runner = TensorRTRunner()

        profile = runner.find_best_profile(input_tensor)
        assert profile == 2


def test_find_best_profile_for_single_input_return_default_profile_with_idx_0():
    shapes = {
        0: {"input__0": [(1, 5), (100000, 5), (200000, 5)]},
        1: {"input__0": [(1, 4), (1, 4), (1, 4)]},
        2: {"input__0": [(200, 4), (200, 4), (250, 4)]},
        3: {"input__0": [(300, 4), (300, 4), (350, 4)]},
    }

    input_tensor = {"input__0": MagicMock(shape=[1, 5])}

    with patch.multiple(
        TensorRTRunner,
        __init__=MagicMock(return_value=None),
        __del__=MagicMock(return_value=None),
        get_num_optimization_profiles=lambda x: 4,
        get_input_metadata=lambda s: [1],
        is_binding_input=lambda s, name: True,
        get_binding_name=lambda s, idx: "input__0",
        get_profile_shape=lambda s, idx, name: shapes[idx][name],
    ):
        runner = TensorRTRunner()

        profile = runner.find_best_profile(input_tensor)
        assert profile == 0
