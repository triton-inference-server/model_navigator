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
"""Test for API configs"""
import pytest

from model_navigator.api.config import (
    DEFAULT_TENSORRT_PRECISION,
    DEFAULT_TENSORRT_PRECISION_MODE,
    CustomConfigForFormat,
    Format,
    JitType,
    OnnxConfig,
    TensorFlowConfig,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TorchConfig,
    TorchTensorRTConfig,
    _custom_configs,
    map_custom_configs,
)


def test_tensorflow_config_has_valid_name_and_format():
    config = TensorFlowConfig()
    assert config.name() == "TensorFlow"
    assert config.format == Format.TF_SAVEDMODEL


def test_tensorflow_tensorrt_config_has_valid_name_and_format():
    config = TensorFlowTensorRTConfig()
    assert config.name() == "TensorFlowTensorRT"
    assert config.format == Format.TF_TRT


def test_tensorflow_tensorrt_config_defaults_reset_values_to_initial():
    config = TensorFlowTensorRTConfig(precision=(TensorRTPrecision.FP32,))
    config.defaults()
    assert config.precision == DEFAULT_TENSORRT_PRECISION


def test_torch_config_has_valid_name_and_format():
    config = TorchConfig()
    assert config.name() == "Torch"
    assert config.format == Format.TORCHSCRIPT


def test_torch_config_has_strict_true_by_default():
    config = TorchConfig()
    assert config.strict is True


def test_torch_config_defaults_reset_values_to_initial():
    config = TorchConfig(strict=False, jit_type=(JitType.TRACE,))
    config.defaults()
    assert config.strict is True
    assert config.jit_type == (JitType.SCRIPT, JitType.TRACE)


def test_torch_tensorrt_config_has_valid_name_and_format():
    config = TorchTensorRTConfig()
    assert config.name() == "TorchTensorRT"
    assert config.format == Format.TORCH_TRT


def test_torch_tensorrt_config_defaults_reset_values_to_initial():
    config = TorchTensorRTConfig(
        precision=(TensorRTPrecision.FP32,),
        precision_mode=TensorRTPrecisionMode.MIXED,
    )
    config.defaults()
    assert config.precision == DEFAULT_TENSORRT_PRECISION
    assert config.precision_mode == DEFAULT_TENSORRT_PRECISION_MODE


def test_onnx_config_has_valid_name_and_format():
    config = OnnxConfig()
    assert config.name() == "Onnx"
    assert config.format == Format.ONNX


def test_tensorrt_config_has_valid_name_and_format():
    config = TensorRTConfig()
    assert config.name() == "TensorRT"
    assert config.format == Format.TENSORRT


def test_tensorrt_config_defaults_reset_values_to_initial():
    config = TensorRTConfig(
        precision=(TensorRTPrecision.FP32,),
        precision_mode=TensorRTPrecisionMode.MIXED,
    )
    config.defaults()
    assert config.precision == DEFAULT_TENSORRT_PRECISION
    assert config.precision_mode == DEFAULT_TENSORRT_PRECISION_MODE


def test_map_custom_configs_return_empty_dict_when_empty_list_pass():
    data = map_custom_configs([])
    assert data == {}


def test_map_custom_configs_return_dict_when_passed_list_of_configs():
    tensorrt_config = TensorRTConfig(
        precision=(TensorRTPrecision.FP32,),
        precision_mode=TensorRTPrecisionMode.MIXED,
    )
    torch_tensorrt_config = TorchTensorRTConfig(
        precision=(TensorRTPrecision.FP32,),
        precision_mode=TensorRTPrecisionMode.MIXED,
    )
    data = map_custom_configs(
        [
            tensorrt_config,
            torch_tensorrt_config,
        ]
    )

    assert len(data) == 2
    assert data[tensorrt_config.name()] == tensorrt_config
    assert data[torch_tensorrt_config.name()] == torch_tensorrt_config


def test_custom_configs_raise_error_when_config_with_duplicated_name_defined():
    class TestFormat(CustomConfigForFormat):
        def format(self):
            return Format.TORCH

        @classmethod
        def name(cls):
            return "Torch"

    with pytest.raises(AssertionError):
        _custom_configs()


def test_custom_configs_raise_error_when_config_with_duplicated_format_defined():
    class TestFormat(CustomConfigForFormat):
        @property
        def format(self):
            return Format.ONNX

        @classmethod
        def name(cls):
            return "FakeONNX"

    with pytest.raises(AssertionError):
        _custom_configs()
