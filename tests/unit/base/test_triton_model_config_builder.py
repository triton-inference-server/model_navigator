# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np

from model_navigator.triton import TensorRTLLMModelConfig
from model_navigator.triton.model_config_builder import ModelConfigBuilder
from model_navigator.triton.specialized_configs import (
    AutoMixedPrecisionAccelerator,
    Backend,
    InputTensorSpec,
    ONNXModelConfig,
    ONNXOptimization,
    OutputTensorSpec,
    PythonModelConfig,
    PyTorchModelConfig,
    TensorFlowModelConfig,
    TensorFlowOptimization,
    TensorRTAccelerator,
    TensorRTModelConfig,
    TensorRTOptimization,
)


def test_from_onnx_config_return_model_config_when_valid_data_passed():
    model_config = ModelConfigBuilder.from_onnx_config(
        model_name="ONNXModel",
        model_version=1,
        onnx_config=ONNXModelConfig(optimization=ONNXOptimization(accelerator=TensorRTAccelerator())),
    )

    assert model_config.model_name == "ONNXModel"
    assert model_config.model_version == 1
    assert model_config.backend == Backend.ONNXRuntime
    assert isinstance(model_config.optimization, ONNXOptimization)


def test_from_pytorch_config_return_model_config_when_valid_data_passed():
    model_config = ModelConfigBuilder.from_pytorch_config(
        model_name="PyTorchModel",
        model_version=1,
        pytorch_config=PyTorchModelConfig(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(-1,)),
                InputTensorSpec(name="INPUT_2", dtype=np.dtype("bytes"), shape=(100, 100)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        ),
    )

    assert model_config.model_name == "PyTorchModel"
    assert model_config.model_version == 1
    assert model_config.backend == Backend.PyTorch
    assert len(model_config.inputs) == 2
    assert len(model_config.outputs) == 1


def test_from_python_config_return_model_config_when_valid_data_passed():
    model_config = ModelConfigBuilder.from_python_config(
        model_name="PythonModel",
        model_version=1,
        python_config=PythonModelConfig(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(-1,)),
                InputTensorSpec(name="INPUT_2", dtype=np.dtype("bytes"), shape=(100, 100)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        ),
    )

    assert model_config.model_name == "PythonModel"
    assert model_config.model_version == 1
    assert model_config.backend == Backend.Python
    assert len(model_config.inputs) == 2
    assert len(model_config.outputs) == 1


def test_from_tensorflow_config_return_model_config_when_valid_data_passed():
    model_config = ModelConfigBuilder.from_tensorflow_config(
        model_name="TensorFlowModel",
        model_version=1,
        tensorflow_config=TensorFlowModelConfig(
            optimization=TensorFlowOptimization(accelerator=AutoMixedPrecisionAccelerator())
        ),
    )

    assert model_config.model_name == "TensorFlowModel"
    assert model_config.model_version == 1
    assert model_config.backend == Backend.TensorFlow
    assert isinstance(model_config.optimization, TensorFlowOptimization)


def test_from_tensorrt_config_return_model_config_when_valid_data_passed():
    model_config = ModelConfigBuilder.from_tensorrt_config(
        model_name="TensorRTModel",
        model_version=1,
        tensorrt_config=TensorRTModelConfig(
            optimization=TensorRTOptimization(
                cuda_graphs=True,
            )
        ),
    )

    assert model_config.model_name == "TensorRTModel"
    assert model_config.model_version == 1
    assert model_config.backend == Backend.TensorRT
    assert isinstance(model_config.optimization, TensorRTOptimization)


def test_from_tensorrt_llm_config_return_model_config_when_valid_data_passed():
    model_config = ModelConfigBuilder.from_tensorrt_llm_config(
        model_name="TensorRTLLMModel",
        model_version=1,
        tensorrt_llm_config=TensorRTLLMModelConfig(),
    )

    assert model_config.model_name == "TensorRTLLMModel"
    assert model_config.model_version == 1
    assert model_config.backend == Backend.TensorRTLLM
    assert model_config.parameters["gpt_model_type"] == "inflight_batching"
    assert model_config.parameters["gpt_model_path"] is None
    assert model_config.parameters["batch_scheduler_policy"] == "max_utilization"
    assert model_config.parameters["FORCE_CPU_ONLY_INPUT_TENSORS"] == "no"
    assert (
        model_config.parameters["executor_worker_path"] == "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"
    )
