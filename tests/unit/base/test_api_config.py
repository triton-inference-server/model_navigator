# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

from model_navigator.configuration import (
    DEFAULT_TENSORRT_PRECISION,
    DEFAULT_TENSORRT_PRECISION_MODE,
    CustomConfigForFormat,
    Format,
    JitType,
    OnnxConfig,
    OptimizationProfile,
    TensorFlowConfig,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TensorRTProfile,
    TorchConfig,
    TorchExportConfig,
    TorchScriptConfig,
    TorchTensorRTConfig,
    _custom_configs,
    map_custom_configs,
)
from model_navigator.configuration.device import (
    get_id_from_device_string,
    validate_device_string,
    validate_device_string_for_cuda,
)
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.inplace.config import OptimizeConfig


def test_tensorrt_config_raise_exception_when_trt_profile_and_trt_profiles_are_both_set():
    with pytest.raises(ModelNavigatorConfigurationError):
        TensorRTConfig(trt_profile=TensorRTProfile(), trt_profiles=[TensorRTProfile()])


def test_tensorflow_tensorrt_config_raise_exception_when_trt_profile_and_trt_profiles_are_both_set():
    with pytest.raises(ModelNavigatorConfigurationError):
        TensorFlowTensorRTConfig(trt_profile=TensorRTProfile(), trt_profiles=[TensorRTProfile()])


def test_torch_tensorrt_config_raise_exception_when_trt_profile_and_trt_profiles_are_both_set():
    with pytest.raises(ModelNavigatorConfigurationError):
        TorchTensorRTConfig(trt_profile=TensorRTProfile(), trt_profiles=[TensorRTProfile()])


def test_tensorrt_based_format_constructs_correctly_with_trt_profile():
    TensorRTConfig(trt_profile=TensorRTProfile())
    TensorFlowTensorRTConfig(trt_profile=TensorRTProfile())
    TorchTensorRTConfig(trt_profile=TensorRTProfile())


def test_tensorrt_based_format_constructs_correctly_with_trt_profiles():
    trt_profiles = [TensorRTProfile(), TensorRTProfile(), TensorRTProfile()]

    TensorRTConfig(trt_profiles=trt_profiles)
    TensorFlowTensorRTConfig(trt_profiles=trt_profiles)
    TorchTensorRTConfig(trt_profiles=trt_profiles)


def test_tensorrt_based_configs_return_valid_profiles():
    trt_profile = TensorRTProfile().add(name="input_0", min=(224, 224, 3), opt=(224, 224, 3), max=(224, 224, 3))

    assert TensorRTConfig(trt_profile=trt_profile).trt_profiles[0] == trt_profile
    assert TensorFlowTensorRTConfig(trt_profile=trt_profile).trt_profiles[0] == trt_profile
    assert TorchTensorRTConfig(trt_profile=trt_profile).trt_profiles[0] == trt_profile

    assert TensorRTConfig(trt_profiles=[trt_profile]).trt_profiles[0] == trt_profile
    assert TensorFlowTensorRTConfig(trt_profiles=[trt_profile]).trt_profiles[0] == trt_profile
    assert TorchTensorRTConfig(trt_profiles=[trt_profile]).trt_profiles[0] == trt_profile


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
    assert config.format == Format.TORCH
    assert config.custom_args is None


def test_torch_config_defaults_reset_values_to_initial():
    config = TorchConfig(inference_mode=False, autocast=False, custom_args={"key": "value"})
    config.defaults()
    assert config.autocast is True
    assert config.inference_mode is True
    assert config.custom_args is None


def test_torch_script_config_has_valid_name_and_format():
    config = TorchScriptConfig()
    assert config.name() == "TorchScript"
    assert config.format == Format.TORCHSCRIPT


def test_torch_script_config_has_strict_true_by_default():
    config = TorchScriptConfig()
    assert config.strict is True


def test_torch_script_config_defaults_reset_values_to_initial():
    config = TorchScriptConfig(strict=False, jit_type=(JitType.TRACE,), inference_mode=False, autocast=False)
    config.defaults()
    assert config.strict is True
    assert config.jit_type == (JitType.SCRIPT, JitType.TRACE)
    assert config.autocast is True
    assert config.inference_mode is True


def test_torch_export_config_has_valid_name_and_format():
    config = TorchExportConfig()
    assert config.name() == "TorchExport"
    assert config.format == Format.TORCH_EXPORTEDPROGRAM


def test_torch__export_config_defaults_reset_values_to_initial():
    config = TorchExportConfig(inference_mode=False, autocast=False)
    config.defaults()
    assert config.autocast is True
    assert config.inference_mode is True


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
    data = map_custom_configs([
        tensorrt_config,
        torch_tensorrt_config,
    ])

    assert len(data) == 2
    assert data[tensorrt_config.name()] == tensorrt_config
    assert data[torch_tensorrt_config.name()] == torch_tensorrt_config


def test_tensorrt_config_raise_error_when_invalid_optimization_level_provided():
    with pytest.raises(
        ModelNavigatorConfigurationError,
        match="TensorRT `optimization_level` must be between 0 and 5. Provided value: 6.",
    ):
        TensorRTConfig(optimization_level=6)

    with pytest.raises(
        ModelNavigatorConfigurationError,
        match="TensorRT `optimization_level` must be between 0 and 5. Provided value: -1.",
    ):
        TensorRTConfig(optimization_level=-1)

    config = TensorRTConfig(optimization_level=2)
    assert config.optimization_level == 2


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


def test_optimization_profile_raise_error_when_windows_size_less_than_1():
    with pytest.raises(ModelNavigatorConfigurationError, match="`window_size` must be greater or equal 1."):
        OptimizationProfile(window_size=0)


def test_optimization_profile_raise_error_when_stabilization_windows_less_than_1():
    with pytest.raises(ModelNavigatorConfigurationError, match="`stabilization_windows` must be greater or equal 1."):
        OptimizationProfile(stabilization_windows=0)


def test_optimization_profile_raise_error_when_min_trials_less_than_1():
    with pytest.raises(ModelNavigatorConfigurationError, match="`min_trials` must be greater or equal 1."):
        OptimizationProfile(min_trials=0)


def test_optimization_profile_raise_error_when_max_trials_less_than_1():
    with pytest.raises(ModelNavigatorConfigurationError, match="`max_trials` must be greater or equal 1."):
        OptimizationProfile(max_trials=0)


def test_optimization_profile_raise_error_when_stability_percentage_equal_0():
    with pytest.raises(ModelNavigatorConfigurationError, match="`stability_percentage` must be greater than 0.0."):
        OptimizationProfile(stability_percentage=0.0)


def test_optimization_profile_raise_error_when_min_trials_less_than_stabilization_windows():
    with pytest.raises(
        ModelNavigatorConfigurationError, match="`min_trials` must be greater or equal than `stabilization_windows`."
    ):
        OptimizationProfile(stabilization_windows=2, min_trials=1)


def test_optimization_profile_raise_error_when_max_trials_less_than_min_trials():
    with pytest.raises(ModelNavigatorConfigurationError, match="`max_trials` must be greater or equal `min_trials`."):
        OptimizationProfile(max_trials=1, min_trials=2, stabilization_windows=1)


def test_validate_device_string_not_raises_exception_for_valid_strings():
    validate_device_string("cpu")
    validate_device_string("cuda")
    for i in range(10):
        validate_device_string(f"cuda:{i}")


def test_validate_device_string_raises_exception_for_incorrect_strings():
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("cuda:1a")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("cuda:1:2")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("cuda:")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("cpu:1,2")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("cpu:1")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("0")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("cpu:1,2")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string("cu")


def test_validate_device_string_for_cuda_not_raises_exception_for_valid_strings():
    validate_device_string_for_cuda("cuda")
    for i in range(10):
        validate_device_string_for_cuda(f"cuda:{i}")


def test_validate_device_string_for_cuda_raises_exception_for_incorrect_strings():
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("cuda:1a")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("cuda:1:2")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("cuda:")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("cuda:1,2")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("0")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("cpu:1,2")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("cu")
    with pytest.raises(ModelNavigatorConfigurationError):
        validate_device_string_for_cuda("cpu")


def test_get_id_from_device_string_returns_device_id_when_available():
    for i in range(10):
        assert get_id_from_device_string(f"cuda:{i}") == i


def test_get_id_from_device_string_returns_none_when_device_id_is_not_available():
    assert get_id_from_device_string("cpu") is None
    assert get_id_from_device_string("cuda") == 0
    assert get_id_from_device_string("cuda:1a") is None
    assert get_id_from_device_string("cuda:1:2") is None
    assert get_id_from_device_string("cuda:1,2") is None
    assert get_id_from_device_string("cuda:1-2") is None
    assert get_id_from_device_string("cuda:1 2") is None
    assert get_id_from_device_string("uda") is None
    assert get_id_from_device_string("cud") is None
    assert get_id_from_device_string("") is None
    assert get_id_from_device_string(" ") is None
    assert get_id_from_device_string("1") is None
    assert get_id_from_device_string("2") is None
    assert get_id_from_device_string("cpu:1") is None
    assert get_id_from_device_string("cuda:") is None


def test_optimization_config_is_cloning_correctly():
    opt_config = OptimizeConfig(
        target_formats=(
            Format.TORCH,
            Format.TORCHSCRIPT,
            Format.ONNX,
            Format.TENSORRT,
        ),
        runners=(
            "TensorRT",
            "TorchCUDA",
            "OnnxCUDA",
            "TorchScriptCUDA",
            "TorchCompileCUDA",
        ),
        optimization_profile=OptimizationProfile(max_batch_size=64),
        custom_configs=[
            TorchConfig(autocast=False, custom_args={"truncate_long_and_double": True}),
            TorchScriptConfig(autocast=False),
            TensorRTConfig(
                precision=(TensorRTPrecision.BF16, TensorRTPrecision.FP16),
                onnx_parser_flags=[1],
            ),
        ],
    )

    cloned_opt_config = opt_config.clone()
    cloned_opt_config.runners = ("TensorRT",)
    cloned_opt_config.optimization_profile.max_batch_size = 32
    cloned_opt_config.custom_configs[0].autocast = True  # pytype: disable=attribute-error
    cloned_opt_config.custom_configs[0].custom_args["truncate_long_and_double"] = (
        False  # pytype: disable=attribute-error
    )

    # nothing changed in original object
    assert len(opt_config.runners) == 5
    assert opt_config.optimization_profile.max_batch_size == 64
    assert opt_config.custom_configs[0].autocast is False  # pytype: disable=attribute-error
    assert (
        opt_config.custom_configs[0].custom_args["truncate_long_and_double"] is True  # pytype: disable=attribute-error
    )
