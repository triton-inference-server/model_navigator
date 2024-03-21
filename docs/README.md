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

# Triton Model Navigator

## Overview

Welcome to the [Triton Model Navigator](https://github.com/triton-inference-server/model_navigator), an inference toolkit designed
for optimizing and deploying Deep Learning models with a focus on NVIDIA GPUs. The Triton Model Navigator streamlines the
process of moving models and pipelines implemented in [PyTorch](https://pytorch.org),
[TensorFlow](https://www.tensorflow.org), and/or [ONNX](https://onnx.ai)
to [TensorRT](https://github.com/NVIDIA/TensorRT).

The Triton Model Navigator automates several critical steps, including model export, conversion, correctness testing, and
profiling. By providing a single entry point for various supported frameworks, users can efficiently search for the best
deployment option using the per-framework optimize function. The resulting optimized models are ready for deployment on
either [PyTriton](https://github.com/triton-inference-server/pytriton)
or [Triton Inference Server](https://github.com/triton-inference-server/server).

## Features at Glance

The distinct capabilities of the Triton Model Navigator are summarized in the feature matrix:

| Feature                     | Description                                                                                                                                      |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Easy-of-use                 | A single line of code to run all possible optimization paths directly from your source code                                                         |
| Wide Framework Support      | Compatible with various machine learning frameworks including PyTorch, TensorFlow, and ONNX                                                      |
| Models Optimization         | Enhance the performance of models such as ResNET and BERT for efficient inference deployment                                                     |
| Pipelines Optimization      | Streamline Python code pipelines for models such as Stable Diffusion and Whisper using Inplace Optimization, exclusive to PyTorch                |
| Model Export and Conversion | Automate the process of exporting and converting models between various formats with focus on TensorRT and Torch-TensorRT                        |
| Correctness Testing         | Ensures the converted model produces correct outputs validating against the original model                                                        |
| Performance Profiling       | Profiles models to select the optimal format based on performance metrics such as latency and throughput to optimize target hardware utilization |
| Models Deployment           | Automates models and pipelines deployment on PyTriton and the Triton Inference Server through a dedicated API                                                          |

## Support Matrix for Frameworks

The Triton Model Navigator efficiently produces various optimized models ready for deployment. The accompanying table
showcases the diverse model formats achievable through the Triton Model Navigator across different frameworks, highlighting its
versatility.

**Table:** Supported conversion target formats per supported Python framework or file.

| **PyTorch**        | **TensorFlow 2**       | **JAX**                | **ONNX** |
|--------------------|------------------------|------------------------|----------|
| Torch Compile      | SavedModel             | SavedModel             | TensorRT |
| TorchScript Trace  | TensorRT in TensorFlow | TensorRT in TensorFlow |          |
| TorchScript Script | ONNX                   | ONNX                   |          |
| Torch-TensorRT     | TensorRT               | TensorRT               |          |
| ONNX               |                        |                        |          |
| TensorRT           |                        |                        |          |

**Note:** The Triton Model Navigator has the capability to support any Python function as input. However, in this particular
case, its role is limited to profiling the function without generating any serialized models.

The Inplace Optimize feature is dedicated for PyTorch to optimize pipelines patching `nn.Modules` and optimize them to
TensorRT. The table below highlights the possible optimization paths for Inplace Optimize:

**Table:** Supported conversion target formats for Inplace Optimize.

| **PyTorch**        |
|--------------------|
| Torch Compile      |
| TorchScript Trace  |
| TorchScript Script |
| Torch-TensorRT     |
| ONNX               |
| TensorRT           |

## What next?

Learn more about using the Triton Model Navigator in [Quick Start](quick_start.md), where you will find more information about
optimizing models and serving inference.
