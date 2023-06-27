<!--
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

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

Model optimization plays a crucial role in unlocking the maximum performance capabilities of the underlying hardware. By
applying various transformation techniques, models can be optimized to fully utilize the specific features offered by
the hardware architecture to improve the inference performance and cost. Furthermore, in many cases allow for
serialization of models, separating them from the source code. The serialization process enhances portability, allowing
the models to be seamlessly deployed in production environments. The decoupling of models from the source code also
facilitates maintenance, updates, and collaboration among developers. However, this process comprises multiple steps and
offers various potential paths, making manual execution complicated and time-consuming.

The [Triton Model Navigator](https://github.com/triton-inference-server/model_navigator) offers a user-friendly and
automated solution for optimizing and deploying machine learning models. Using a single entry point for
various supported frameworks, allowing users to start the process of searching for the best deployment option with a
single call to the dedicated `optimize` function. Model Navigator handles model export, conversion, correctness testing,
and profiling to select optimal model format and save generated artifacts for inference deployment on the
[PyTriton](https://github.com/triton-inference-server/pytriton)
or [Triton Inference Server](https://github.com/triton-inference-server/server).

The high-level flowchart below illustrates the process of moving models from source code to deployment optimized formats
with the support of the Model Navigator.

![Overview](assets/overview.svg)

## Support Matrix

The Model Navigator generates multiple optimized and production-ready models. The table below illustrates the model
formats that can be obtained by using the Model Navigator with various frameworks.

**Table:** Supported conversion target formats per each supported Python framework or file.

| **PyTorch**        | **TensorFlow 2**       | **JAX**                | **ONNX** |
|--------------------|------------------------|------------------------|----------|
| Torch Compile      | SavedModel             | SavedModel             | TensorRT |
| TorchScript Trace  | TensorRT in TensorFlow | TensorRT in TensorFlow |          |
| TorchScript Script | ONNX                   | ONNX                   |          |
| Torch-TensorRT     | TensorRT               | TensorRT               |          |
| ONNX               |                        |                        |          |
| TensorRT           |                        |                        |          |

**Note:** The Model Navigator has the capability to support any Python function as input. However, in this particular
case, its role is limited to profiling the function without generating any serialized models.

The Model Navigator stores all artifacts within the `navigator_workspace`. Additionally, it provides an option to save
a portable and transferable `Navigator Package` - an artifact that includes only the models with minimal latency and
maximal throughput. This package also includes base formats that can be used to regenerate the `TensorRT` plan on the
target hardware.

**Table:** Model formats that can be generated from saved `Navigator Package` and from model sources.

| **From model source** | **From Navigator Package** |
|-----------------------|----------------------------|
| SavedModel            | TorchTensorRT              |
| TensorFlowTensorRT    | TensorFlowTensorRT         |
| TorchScript Trace     | ONNX                       |
| TorchScript Script    | TensorRT                   |
| Torch 2 Compile       |                            |
| TorchTensorRT         |                            |
| ONNX                  |                            |
| TensorRT              |                            |

## What next?

Learn more about using Model Navigator in [quick start](quick_start.md) where you will find more information about
optimizing models and serving inference.
