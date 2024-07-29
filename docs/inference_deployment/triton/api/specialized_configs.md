<!--
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Specialized Configs for Triton Backends

The Python API provides specialized configuration classes that help provide only
available options for the given type of model.

::: model_navigator.triton.BaseSpecializedModelConfig

::: model_navigator.triton.ONNXModelConfig
::: model_navigator.triton.ONNXOptimization

::: model_navigator.triton.PythonModelConfig

::: model_navigator.triton.PyTorchModelConfig

::: model_navigator.triton.TensorFlowModelConfig
::: model_navigator.triton.TensorFlowOptimization

::: model_navigator.triton.TensorRTModelConfig
::: model_navigator.triton.TensorRTOptimization

::: model_navigator.triton.TensorRTLLMModelConfig
::: model_navigator.triton.BatchingStrategy
::: model_navigator.triton.BatchSchedulerPolicy
::: model_navigator.triton.DecodingMode
::: model_navigator.triton.KVCacheConfig
::: model_navigator.triton.PeftCacheConfig
