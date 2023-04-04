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

# Installation

This section describe how to install the tool. We assume you are comfortable with Python programming language
and familiar with Machine Learning models.

## Prerequisites

The following prerequisites must be fulfilled to use Triton Model Navigator

- Installed Python `3.8+`
- Installed [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for TensorRT models export.

We recommend to use NGC Containers for PyTorch and TensorFlow which provide have all necessary dependencies:

- [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

The library can be installed in:

- system environment
- virtualenv
- [Docker](https://www.docker.com/)

The NVIDIA optimized Docker images for Python frameworks could be obtained
from [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers).

For using NVIDIA optimized Docker images we recommend to
install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to
run model inference on NVIDIA GPU.

## Installation
The package can be installed from `pypi.org` using extra index url:


```shell
pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[<extras,>]
```

or with nvidia-pyindex:

```shell
pip install nvidia-pyindex
pip install -U triton-model-navigator[<extras,>]
```

To install Triton Model Navigator from source use pip command:

```shell
$ pip install --extra-index-url https://pypi.ngc.nvidia.com .[<extras,>]
```

Extras:

- `tensorflow` - Model Navigator with dependencies for TensorFlow2
- `jax` - Model Navigator with dependencies for JAX

For using with PyTorch no extras are needed.

## Building the wheel

The Triton Model Navigator can be built as wheel. On that purpose the Makefile provide necessary commands.

The first is required to install necessary packages to perform build.
```
make install-dev
```

Once the environment contain required packages run:
```shell
make dist
```

The wheel is going to be generated in `dist` catalog.
