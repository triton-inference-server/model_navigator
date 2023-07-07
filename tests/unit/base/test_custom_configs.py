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

from model_navigator.api.config import (
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
)
from model_navigator.core.constants import DEFAULT_MAX_WORKSPACE_SIZE, DEFAULT_MIN_SEGMENT_SIZE, DEFAULT_ONNX_OPSET


def test_default_values_for_tensorflow_custom_config():
    tensorflow_config = TensorFlowConfig()
    assert tensorflow_config.jit_compile == (None,)
    assert tensorflow_config.enable_xla == (None,)
    assert tensorflow_config.format == Format.TF_SAVEDMODEL


def test_default_values_for_tensorflow_tensorrt_custom_config():
    tensorflow_tensorrt_config = TensorFlowTensorRTConfig()
    assert tensorflow_tensorrt_config.precision == (
        TensorRTPrecision.FP32,
        TensorRTPrecision.FP16,
    )
    assert tensorflow_tensorrt_config.max_workspace_size == DEFAULT_MAX_WORKSPACE_SIZE
    assert tensorflow_tensorrt_config.minimum_segment_size == DEFAULT_MIN_SEGMENT_SIZE
    assert tensorflow_tensorrt_config.format == Format.TF_TRT


def test_default_values_for_torch_custom_config():
    torch_config = TorchConfig()
    assert torch_config.jit_type == (JitType.SCRIPT, JitType.TRACE)
    assert torch_config.format == Format.TORCHSCRIPT


def test_default_values_for_torch_tensorrt_custom_confg():
    torch_tensorrt_config = TorchTensorRTConfig()
    assert torch_tensorrt_config.precision == (
        TensorRTPrecision.FP32,
        TensorRTPrecision.FP16,
    )
    assert torch_tensorrt_config.precision_mode == TensorRTPrecisionMode.HIERARCHY
    assert torch_tensorrt_config.max_workspace_size == DEFAULT_MAX_WORKSPACE_SIZE
    assert torch_tensorrt_config.format == Format.TORCH_TRT


def test_default_values_for_onnx_config():
    onnx_config = OnnxConfig()
    assert onnx_config.opset == DEFAULT_ONNX_OPSET
    assert onnx_config.dynamic_axes is None
    assert onnx_config.onnx_extended_conversion is False
    assert onnx_config.format == Format.ONNX


def test_default_values_for_tensorrt_config():
    tensorrt_config = TensorRTConfig()
    assert tensorrt_config.precision == (
        TensorRTPrecision.FP32,
        TensorRTPrecision.FP16,
    )
    assert tensorrt_config.precision_mode == TensorRTPrecisionMode.HIERARCHY
    assert tensorrt_config.trt_profiles is None
    assert tensorrt_config.max_workspace_size == DEFAULT_MAX_WORKSPACE_SIZE
    assert tensorrt_config.format == Format.TENSORRT
