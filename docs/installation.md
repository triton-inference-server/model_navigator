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

# Installation

## Prerequisites

Before proceeding with the installation of the Triton Model Navigator, ensure your system meets the following criteria:

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: Version `3.9` or newer
- NVIDIA GPU

You can use NGC Containers for PyTorch and TensorFlow which contain all necessary dependencies:

- [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

The library can be installed in:

- system environment
- virtualenv
- [Docker](https://www.docker.com/)

The NVIDIA optimized Docker images for Python frameworks could be obtained
from [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers).

For using NVIDIA optimized Docker images, we recommend installing
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to
run model inference on NVIDIA GPU.

## Install

The Triton Model Navigator can be installed from `pypi.org`.

### Installing with PyTorch extras

For installing with PyTorch dependencies, use:

```shell
pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[torch]
```

or with nvidia-pyindex:

```shell
pip install nvidia-pyindex
pip install -U triton-model-navigator[torch]
```

### Installing with TensorFlow extras

For installing with TensorFlow dependencies, use:

```shell
pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[tensorflow]
```

or with nvidia-pyindex:

```shell
pip install nvidia-pyindex
pip install -U triton-model-navigator[tensorflow]
```

### Installing with JAX extras (experimental)

For installing with JAX dependencies, use:

```shell
pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[jax]
```

or with nvidia-pyindex:

```shell
pip install nvidia-pyindex
pip install -U triton-model-navigator[jax]
```

### Installing with onnxruntime-gpu for CUDA 11

The default CUDA version for ONNXRuntime since 1.19.0 is CUDA 12. To install with CUDA 11 support use following extra index url:
```shell
.. --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ ..
```

## Building the wheel

The Triton Model Navigator can be built as a wheel. We have prepared all necessary steps under `Makefile` command.

Firstly, install the Triton Model Navigator with development packages:
```shell
make install-dev
```

Next, simply run:

```shell
make dist
```

The wheel will be generated in the `dist` catalog.
