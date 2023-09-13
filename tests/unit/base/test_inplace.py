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
"""Tests for ConvertONNX2TRT conversion.

Note:
     Those test do not execute the conversion.
     The tests are checking if correct paths are executed on input arguments.
"""
from unittest.mock import MagicMock

import pytest

from model_navigator.inplace.config import Mode, OptimizeConfig, inplace_config
from model_navigator.inplace.model import OptimizedModule, PassthroughModule, RecordModule
from model_navigator.inplace.registry import module_registry
from model_navigator.inplace.utils import get_object_name


def test_get_object_name():
    assert get_object_name(MagicMock()) == "unittest.mock.MagicMock"


def test_config_parse_mode_str():
    inplace_config.mode = "optimize"
    assert inplace_config.mode == Mode.OPTIMIZE


def test_config_raise_error_on_invalid_num_samples():
    with pytest.raises(ValueError):
        inplace_config.min_num_samples = -1


def test_model_registry_check_all_ready_returns_true_when_all_models_ready():
    module_registry._registry = {
        "model1": MagicMock(is_optimized=False, is_ready_for_optimization=True),
        "model2": MagicMock(is_optimized=True, is_ready_for_optimization=False),
    }
    assert module_registry.check_all_ready()


def test_model_registry_check_all_ready_returns_false_when_not_all_models_ready():
    module_registry._registry = {
        "model1": MagicMock(is_optimized=False, is_ready_for_optimization=False),
        "model2": MagicMock(is_optimized=True, is_ready_for_optimization=True),
    }
    assert not module_registry.check_all_ready()


def test_pass_model_is_optimized_returns_false():
    module = PassthroughModule(
        module=MagicMock(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_optimized


def test_optimized_model_is_optimized_returns_true(mocker):
    with mocker.patch.object(OptimizedModule, "__init__", return_value=None):
        module = OptimizedModule(
            module=MagicMock(),
            name="model_name",
            input_mapping=lambda x: x,
            output_mapping=lambda x: x,
            optimize_config=OptimizeConfig(),
        )
        assert module.is_optimized


def test_recording_model_is_optimized_returns_false():
    module = RecordModule(
        module=MagicMock(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_optimized


def test_pass_model_is_ready_for_optimization_returns_false():
    module = PassthroughModule(
        module=MagicMock(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_ready_for_optimization


def test_recording_model_is_ready_for_optimization_returns_false_when_not_enough_samples():
    module = RecordModule(
        module=MagicMock(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_ready_for_optimization


def test_recording_model_is_ready_for_optimization_returns_true_when_enough_samples():
    module = RecordModule(
        module=MagicMock(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    module._samples_shapes = {"hash": list(range(inplace_config.min_num_samples))}
    assert module.is_ready_for_optimization
