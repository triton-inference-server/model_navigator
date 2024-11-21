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
from model_navigator.configuration import (
    DeviceKind,
    Format,
    JitType,
    OptimizationProfile,
    TensorRTPrecision,
    TensorRTPrecisionMode,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.constants import DEFAULT_MAX_WORKSPACE_SIZE, DEFAULT_ONNX_OPSET
from model_navigator.configuration.model.model_config import (
    ONNXModelConfig,
    TensorFlowSavedModelConfig,
    TensorRTModelConfig,
    TorchModelConfig,
    TorchScriptModelConfig,
)
from model_navigator.frameworks import Framework
from model_navigator.pipelines.builders.find_device_max_batch_size import find_device_max_batch_size_builder
from model_navigator.runners.onnx import OnnxrtCUDARunner
from model_navigator.runners.tensorflow import TensorFlowSavedModelCUDARunner
from model_navigator.runners.torch import TorchCompileCUDARunner, TorchCUDARunner


def test_find_device_max_batch_size_builder_return_execution_unit_when_torch_framework_is_used():
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.TORCH: [
            TorchModelConfig(
                autocast=True,
                inference_mode=True,
                custom_args={
                    "truncate_long_and_double": False,
                    "backend": "torch_tensorrt",
                },
            )
        ],
        Format.TORCHSCRIPT: [
            TorchScriptModelConfig(
                jit_type=JitType.TRACE,
                strict=True,
                autocast=False,
                inference_mode=True,
            )
        ],
        Format.ONNX: [
            ONNXModelConfig(
                opset=DEFAULT_ONNX_OPSET, dynamic_axes={}, dynamo_export=False, graph_surgeon_optimization=True
            )
        ],
        Format.TENSORRT: [
            TensorRTModelConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                optimization_level=None,
                compatibility_level=None,
            )
        ],
    }
    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 3

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == TorchCUDARunner

    configuration = execution_unit.kwargs["configurations"][1]
    assert configuration.runner_cls == TorchCompileCUDARunner

    configuration = execution_unit.kwargs["configurations"][2]
    assert configuration.runner_cls == OnnxrtCUDARunner


def test_find_device_max_batch_size_builder_return_execution_unit_when_tensorflow_framework_is_used():
    config = CommonConfig(
        framework=Framework.TENSORFLOW,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.TF_SAVEDMODEL: [TensorFlowSavedModelConfig(enable_xla=False, jit_compile=False)],
        Format.ONNX: [
            ONNXModelConfig(
                opset=DEFAULT_ONNX_OPSET, dynamic_axes={}, dynamo_export=False, graph_surgeon_optimization=True
            )
        ],
        Format.TENSORRT: [
            TensorRTModelConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                optimization_level=None,
                compatibility_level=None,
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 2

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == TensorFlowSavedModelCUDARunner

    configuration = execution_unit.kwargs["configurations"][1]
    assert configuration.runner_cls == OnnxrtCUDARunner


def test_find_device_max_batch_size_builder_return_execution_unit_when_jax_framework_is_used():
    config = CommonConfig(
        framework=Framework.JAX,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.TF_SAVEDMODEL: [TensorFlowSavedModelConfig(enable_xla=True, jit_compile=True)],
        Format.ONNX: [
            ONNXModelConfig(
                opset=DEFAULT_ONNX_OPSET, dynamic_axes={}, dynamo_export=False, graph_surgeon_optimization=True
            )
        ],
        Format.TENSORRT: [
            TensorRTModelConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                optimization_level=None,
                compatibility_level=None,
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 2

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == TensorFlowSavedModelCUDARunner

    configuration = execution_unit.kwargs["configurations"][1]
    assert configuration.runner_cls == OnnxrtCUDARunner


def test_find_device_max_batch_size_builder_return_execution_unit_when_onnx_framework_is_used():
    config = CommonConfig(
        framework=Framework.ONNX,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.ONNX: [
            ONNXModelConfig(
                opset=DEFAULT_ONNX_OPSET, dynamic_axes={}, dynamo_export=False, graph_surgeon_optimization=True
            )
        ],
        Format.TENSORRT: [
            TensorRTModelConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                optimization_level=None,
                compatibility_level=None,
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 1

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == OnnxrtCUDARunner
