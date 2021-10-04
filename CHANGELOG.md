<!--
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

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

# Changelog

## unreleased
- Fixed `model_repository` path to be  not relative to `<navigator_workspace>` dir
- Handle exit codes correctly from CLI commands
- Conversion related
  - Added support for precision modes to support multiple precision during conversion to TensorRT
  - Fixed nan values appearing in relative tolerance during conversion to TensorRT
- Configuration Model on Triton related
  - Removed default value for `engine_count_per_device`
  - Added possibility to define Triton Custom Backend parameters with `triton_backend_parameters` command
  - Added possibility to define max workspace size for TensorRT backend accelerator using argument `tensorrt_max_workspace_size`
- Profiling related
  - Added `config_search` prefix  to all profiling parameters (BREAKING CHANGE)
  - Added `config_search_max_preferred_batch_size` parameter
  - Added `config_search_backend_parameters` parameter
- fixes and improvements

## 0.2.2 (2021-09-06)

- Updated NVIDIA containers defaults to 21.08

[//]: <> (put here on external component update with short summary what change or link to changelog)
- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 1.7.0
    - [Triton Inference Server Client](https://github.com/triton-inference-server/client/): 2.13.0
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.31.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.11
    - tf2onnx: [v1.9.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.1) (support for ONNX opset 14, tf 1.15 and 2.5)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - possible to define a single profile for TensorRT

## 0.2.1 (2021-08-17)

- Fixed triton-model-config error when tensorrt_capture_cuda_graph flag is not passed
- [Dump Conversion Comparator inputs and outputs into JSON files](docs/conversion.md#verification-of-conversion-Correctness)
- Added information in logs on the tolerance parameters values to pass the conversion verification
- Use `count_windows` mode as default option for Perf Analyzer
- Added possibility to define custom docker images
- Bugfixes

[//]: <> (put here on external component update with short summary what change or link to changelog)
- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 1.6.0
    - [Triton Inference Server Client](https://github.com/triton-inference-server/client/): 2.12.0
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.31.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.11
    - tf2onnx: [v1.9.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.1) (support for ONNX opset 14, tf 1.15 and 2.5)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - possible to define a single profile for TensorRT
    - TensorRT backend acceleration not supported for ONNX Runtime in Triton Inference Server ver. 21.07

## 0.2.0 (2021-07-05)

- comprehensive refactor of command-line API in order to provide more gradual
  pipeline steps execution

[//]: <> (put here on external component update with short summary what change or link to changelog)
- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 21.05
    - tf2onnx: [v1.8.5](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.8.5) (support for ONNX opset 13, tf 1.15 and 2.5)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - issues with TorchScript -> ONNX conversion due to [issue in PyTorch 1.8](https://github.com/pytorch/pytorch/issues/53506)
      - affected NVIDIA PyTorch containers: 20.12, 21.02, 21.03
      - workaround: use PyTorch containers newer than 21.03
    - possible to define a single profile for TensorRT


## 0.1.1 (2021-04-12)
- documentation update

## 0.1.0 (2021-04-09)
- Release of main components:
    - Model Converter - converts the model to a set of variants optimized for inference or to be later optimized by Triton Inference Server backend.
    - Model Repo Builder - setup Triton Inference Server Model Repository, including its configuration.
    - Model Analyzer - select optimal Triton Inference Server configuration based on models compute and memory requirements,
    available computation infrastructure, and model application constraints.
    - Helm Chart Generator - deploy Triton Inference Server and model with optimal configuration to cloud.

- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 21.03+616e8a30
    - tf2onnx: [v1.8.4](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.8.4) (support for ONNX opset 13, tf 1.15 and 2.4)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
    Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    for a detailed summary.

- Known issues
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - issues with TorchScript -> ONNX conversion due to [issue in PyTorch 1.8](https://github.com/pytorch/pytorch/issues/53506)
      - affected NVIDIA PyTorch containers: 20.12, 21.03
      - workaround: use containers different from above
    - Triton Inference Server stays in the background when the profile process is interrupted by the user
