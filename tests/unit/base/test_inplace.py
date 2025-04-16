# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from model_navigator.configuration import (
    TensorRTConfig,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TorchConfig,
)
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.inplace import _initialize_pipeline
from model_navigator.inplace.config import DEFAULT_CACHE_DIR, OptimizeConfig, inplace_cache_dir
from model_navigator.inplace.model import EagerModule, OptimizedModule, RecordingModule
from model_navigator.inplace.registry import ModuleRegistry, module_registry
from model_navigator.inplace.utils import get_object_name
from model_navigator.inplace.wrapper import Module, module
from model_navigator.reporting.optimize.events import OptimizeEvent
from tests.unit.base.mocks.fixtures import mock_event_emitter  # noqa: F401


@pytest.fixture(autouse=True)
def clean_up_registry():
    """Clears registry after test case."""
    yield
    module_registry.clear()


def get_test_module():
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x

    return TestModule()


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


def test_model_registry_check_all_ready_returns_true_when_all_models_ready(clean_up_registry):
    module_registry.register("model1", MagicMock(is_optimized=False, is_ready_for_optimization=True))
    module_registry.register("model2", MagicMock(is_optimized=True, is_ready_for_optimization=False))
    assert module_registry.check_all_ready()


def test_model_registry_check_all_ready_returns_false_when_not_all_models_ready(clean_up_registry):
    module_registry.register("model1", MagicMock(is_optimized=False, is_ready_for_optimization=False))
    module_registry.register("model2", MagicMock(is_optimized=True, is_ready_for_optimization=True))
    assert not module_registry.check_all_ready()


def test_registry_should_emit_events(mock_event_emitter):  # noqa: F811
    # given
    module = MagicMock()
    module.is_optimized = False
    registry = ModuleRegistry()
    registry.event_emitter = mock_event_emitter
    # when
    registry.register("test_module", module)
    registry.optimize()
    registry.clear()
    # then
    events = mock_event_emitter.history
    assert len(events) == 5
    assert events[0] == (
        OptimizeEvent.MODULE_REGISTERED,
        (),
        {
            "name": "test_module",
            "num_modules": 0,
            "num_params": 0,
        },
    )
    assert events[1] == (OptimizeEvent.INPLACE_STARTED, (), {})
    assert events[2] == (
        OptimizeEvent.MODULE_PICKED_FOR_OPTIMIZATION,
        (),
        {
            "name": "test_module",
        },
    )
    assert events[3] == (OptimizeEvent.INPLACE_FINISHED, (), {})
    assert events[4] == (OptimizeEvent.MODULE_REGISTRY_CLEARED, (), {})


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_pass_model_is_optimized_returns_false():
    module = EagerModule(
        module=get_test_module(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_optimized


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_optimized_model_is_optimized_returns_true(mocker):
    mocker.patch.object(OptimizedModule, "__init__", return_value=None)

    module = OptimizedModule(
        module=get_test_module(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert module.is_optimized


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_recording_model_is_optimized_returns_false():
    module = RecordingModule(
        module=get_test_module(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_optimized


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_pass_model_is_ready_for_optimization_returns_false():
    module = EagerModule(
        module=get_test_module(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_ready_for_optimization


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_recording_model_is_ready_for_optimization_returns_false_when_not_enough_samples():
    module = RecordingModule(
        module=get_test_module(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert not module.is_ready_for_optimization


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_recording_model_is_ready_for_optimization_returns_true_when_enough_samples():
    from model_navigator.inplace.config import inplace_config

    module = RecordingModule(
        module=get_test_module(),
        name="model_name",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    module._samples_shapes = {"hash": list(range(inplace_config.min_num_samples))}
    assert module.is_ready_for_optimization


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_record_module_raise_exception_when_invalid_module_wrapped():
    module = MagicMock()
    with pytest.raises(ModelNavigatorUserInputError, match="Only torch modules are supported."):
        RecordingModule(
            module=module,
            name="model1",
            input_mapping=lambda x: x,
            output_mapping=lambda x: x,
            optimize_config=OptimizeConfig(),
        )


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_record_module_alt_forward():
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return torch.Tensor([1])

        def encode(self, x):
            return torch.Tensor([2])

    module = TestModule()

    module1 = RecordingModule(
        module=module,
        name="model1",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    assert module1(torch.Tensor([0])) == torch.ones(1)

    module1 = RecordingModule(
        module=module,
        name="model2",
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        forward=module.encode,
        optimize_config=OptimizeConfig(),
    )
    assert module1(torch.Tensor([0])) == torch.Tensor([2])


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_raise_exception_when_invalid_module_wrapped():
    module = MagicMock()
    with pytest.raises(ModelNavigatorUserInputError, match="Only torch modules are supported."):
        Module(
            module=module,
            name="model1",
        )


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_wrapper_alt_forward(mocker):
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + torch.Tensor([2])

        def encode(self, x):
            return self.forward(x)

    module = TestModule()

    spy_forward = mocker.spy(module, "forward")

    module = Module(module, "model3", forward_func="forward")

    assert module.encode(torch.zeros(1)) == torch.Tensor([2])
    assert spy_forward.call_count == 1


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


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_tags_should_override_config(clean_up_registry):
    # given
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    config = OptimizeConfig(batching=False)
    config_copy = config.clone()

    model_a = Module(TestModule(), name="model_a", precision="fp16")
    model_b = Module(TestModule(), name="model_b", precision="fp32", batching=True)
    model_c = Module(TestModule(), name="model_c")

    # when
    model_a.optimize_config = config
    model_b.optimize_config = config
    model_c.optimize_config = config

    # then
    assert_correct_overrides_in_configs(config, config_copy, model_a, model_b, model_c)


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_factory_tags_should_override_config(clean_up_registry):
    # given
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    custom_configs = (TorchConfig(),)
    config = OptimizeConfig(batching=False, custom_configs=custom_configs)
    config_copy = config.clone()

    @module(name="model_a", precision="fp16")
    def module_a():
        return TestModule()

    @module(name="model_b", precision="fp32", batching=True)
    def module_b():
        return TestModule()

    @module(name="model_c")
    def module_c():
        return TestModule()

    model_a = module_a()
    model_b = module_b()
    model_c = module_c()

    # when
    model_a.optimize_config = config
    model_b.optimize_config = config
    model_c.optimize_config = config

    # then
    assert_correct_overrides_in_configs(config, config_copy, model_a, model_b, model_c)


def assert_correct_overrides_in_configs(config, config_copy, model_a, model_b, model_c):
    # then
    assert config == config_copy, "Config must not be altered by any module"
    assert not model_a.optimize_config.batching
    custom_config = cast(TensorRTConfig, model_a.optimize_config.custom_configs[-1])
    assert custom_config.precision == (TensorRTPrecision.FP16,)
    assert custom_config.precision_mode == TensorRTPrecisionMode.HIERARCHY

    assert model_b.optimize_config.batching
    custom_config = cast(TensorRTConfig, model_b.optimize_config.custom_configs[-1])
    assert custom_config.precision == (TensorRTPrecision.FP32, TensorRTPrecision.FP16)
    assert custom_config.precision_mode == TensorRTPrecisionMode.HIERARCHY

    assert not model_c.optimize_config.batching


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_module_tags_should_partially_override_config(clean_up_registry):
    # given
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    custom_configs = (TensorRTConfig(precision="fp32", precision_mode="single"),)
    config = OptimizeConfig(batching=False, custom_configs=custom_configs)

    model_a = Module(TestModule(), name="model_a", precision="fp16")
    model_b = Module(TestModule(), name="model_b", precision="fp32", batching=True)

    # when
    model_a.optimize_config = config
    model_b.optimize_config = config

    # then
    assert not model_a.optimize_config.batching
    assert model_a.optimize_config.custom_configs == config.custom_configs

    assert model_b.optimize_config.batching
    assert model_b.optimize_config.custom_configs == config.custom_configs


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_initialize_pipeline_not_call_to_method_when_only_one_module_wrapped(mocker):
    # given
    import torch  # pytype: disable=import-error

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    module = TestModule()

    spy_to_method = mocker.spy(module, "to")

    module = Module(module, name="test")

    result = _initialize_pipeline(func=module, model_key="torch", runner_name="TorchCUDA", device="cpu")

    assert result is False
    assert spy_to_method.call_count == 0


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_initialize_pipeline_not_call_to_method_when_more_then_one_module_wrapped_but_not_to_method_in_pipe():
    # given
    import torch  # pytype: disable=import-error

    class TestModule1(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class TestModule2(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class Pipe:
        def __init__(self):
            super().__init__()
            self.module1 = TestModule1()
            self.module2 = TestModule2()

        def __call__(self, *args, **kwargs):
            pass

    pipe = Pipe()

    pipe.module1 = Module(pipe.module1, name="test1")
    pipe.module2 = Module(pipe.module2, name="test2")

    result = _initialize_pipeline(func=pipe, model_key="torch", runner_name="TorchCUDA", device="cpu")
    assert result is False


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_initialize_pipeline_not_call_to_method_when_python_eager_passed(mocker):
    # given
    import torch  # pytype: disable=import-error

    class TestModule1(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class TestModule2(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class Pipe:
        def __init__(self):
            self.module1 = TestModule1()
            self.module2 = TestModule2()

        def __call__(self, *args, **kwargs):
            pass

        def to(self, device):
            self.module1.to(device)
            self.module2.to(device)

    pipe = Pipe()

    spy_to_method = mocker.spy(pipe, "to")

    pipe.module1 = Module(pipe.module1, name="module1")
    pipe.module2 = Module(pipe.module2, name="module2")

    result = _initialize_pipeline(func=pipe, model_key="python", runner_name="eager", device="cpu")

    assert result is False
    assert spy_to_method.call_count == 0


@pytest.mark.skipif(not find_spec("torch"), reason="PyTorch is not installed.")
def test_initialize_pipeline_called_when_more_then_one_module_wrapped_and_pipe_has_to_method(mocker):
    # given
    import torch  # pytype: disable=import-error

    class TestModule1(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class TestModule2(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class Pipe:
        def __init__(self):
            self.module1 = TestModule1()
            self.module2 = TestModule2()

        def __call__(self, *args, **kwargs):
            pass

        def to(self, device):
            self.module1.to(device)
            self.module2.to(device)

    pipe = Pipe()

    spy_to_method = mocker.spy(pipe, "to")

    pipe.module1 = Module(pipe.module1, name="module1")
    pipe.module2 = Module(pipe.module2, name="module2")

    result = _initialize_pipeline(func=pipe, model_key="torch", runner_name="TorchCUDA", device="cpu")

    assert result is True
    assert spy_to_method.call_count == 1
