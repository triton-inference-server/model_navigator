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

import os
import pathlib
from importlib.util import find_spec
from unittest.mock import MagicMock, patch

import pytest

from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.inplace.config import DEFAULT_CACHE_DIR, OptimizeConfig, inplace_cache_dir
from model_navigator.inplace.model import OptimizedModule, PassthroughModule, RecordModule
from model_navigator.inplace.registry import module_registry
from model_navigator.inplace.utils import get_object_name
from model_navigator.inplace.wrapper import Module


def test_get_object_name():
    assert get_object_name(MagicMock()) == "unittest.mock.MagicMock"


@patch.dict(os.environ, {"MODEL_NAVIGATOR_DEFAULT_CACHE_DIR": "/tmp/model_navigator"})
def test_inplace_cache_dir_return_env_variable_value():
    cache_dir = inplace_cache_dir()
    assert cache_dir == pathlib.Path("/tmp/model_navigator")


def test_inplace_config_cache_dir_return_default_value():
    from model_navigator.inplace.config import inplace_config

    assert inplace_config.cache_dir == pathlib.Path(DEFAULT_CACHE_DIR)


def test_config_raise_error_on_invalid_num_samples():
    from model_navigator.inplace.config import inplace_config

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
    from model_navigator.inplace.config import inplace_config

    module = RecordModule(
        module=MagicMock(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    module._samples_shapes = {"hash": list(range(inplace_config.min_num_samples))}
    assert module.is_ready_for_optimization


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_record_module_alt_forward():
    import torch  # pytype: disable=import-error

    module = MagicMock()
    module.side_effect = lambda _: torch.Tensor([1])
    module.forward.side_effect = lambda _: torch.Tensor([2])

    module1 = RecordModule(module=module, name="model1", input_mapping=lambda x: x, output_mapping=lambda x: x)
    assert module1(torch.Tensor([0])) == torch.ones(1)

    module1 = RecordModule(
        module=module, name="model2", input_mapping=lambda x: x, output_mapping=lambda x: x, forward=module.forward
    )
    assert module1(torch.Tensor([0])) == torch.Tensor([2])


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_wrapper_alt_forward():
    import torch  # pytype: disable=import-error

    module = MagicMock()
    # torch __call__ triggers forward, so we need to mock both, and be careful
    module.side_effect = lambda _: module.forward(0) + 1000
    module.forward.side_effect = lambda _: torch.Tensor([2])

    module = Module(module, "model3", forward_func="forward")

    assert module(torch.zeros(1)) == torch.Tensor([2])


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_wrapper_torch_object():
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    module = TestModule()
    nav_module = Module(module, "model4", forward_func="forward")

    assert nav_module(torch.Tensor([1000])) == torch.Tensor([1001])


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_wrapper_complain_on_missing_custom_func_name():
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    with pytest.raises(ModelNavigatorUserInputError):
        Module(TestModule(), "model5", forward_func="non_existing_func")
