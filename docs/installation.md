<!--
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

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

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Requirements](#requirements)
- [Export From Source](#export-from-source)
- [Optimize for Triton Inference Server](#optimize-for-triton-inference-server)
- [Using Docker Container](#using-docker-container)
- [Installing from the Source](#installing-from-the-source)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Requirements
To use Model Navigator Export API you have to have PyTorch or TensorFlow2 already installed on your system.
To export models to TensorRT you have to have [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) installed.

NGC Containers are the recommended environments for Model Navigator Export API, they have all required dependencies:
- [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

The minimal required `Python` version for Triton Model Navigator is `3.8`.

For JAX models, the apropriate JAX library version is required `(CPU, CUDA, TPU)` and all other derived frameworks used by model `(Flax, Haiku)`.

Installation details:
- [JAX](https://github.com/google/jax#installation)
- [Flax](https://github.com/google/flax#quick-install)
- [Haiku](https://github.com/deepmind/dm-haiku#installation)

For JAX models set `XLA_PYTHON_CLIENT_PREALLOCATE` environment variable to avoid Out of Memory issues:

```shell
$ export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

For JAX and TensorFlow2 models, enable tensorflow memory growth to avoid allocating all GPU memory:

```python
import tensorflow

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)
```

## Export From Source

Triton Model Navigator Export API is installed with Model Navigator package.
To install Model Navigator Export API without Model Navigator dependencies use commands:

```shell
$ pip install --extra-index-url https://pypi.ngc.nvidia.com git+https://github.com/triton-inference-server/model_navigator.git@v0.3.3#egg=model-navigator[<extras,>] --upgrade
```

Extras:
- pyt - Model Navigator Export API for PyTorch
- tf - Model Navigator Export API for TensorFlow2
- jax - Model Navigator Export API for JAX.
- cli - Model Navigator CLI
- huggingface - Model Navigator Export API for HuggingFace

## Optimize for Triton Inference Server

Triton Model Navigator Optimize step performs a search of optimal format and deployment configuration
of model or package produced by Export API on Triton Inference Server. The `optimize` command performs
a conversion to available formats, apply additional Triton backends optimizations and use `Triton Model Analyzer` for
profiling looking for best deployment configuration.

## Using Docker Container
The recommended way of using the optimize step is to build a Docker container with all necessary dependencies:
```shell
$ make docker
```

Running the Triton Model Navigator inside the container requires additional volumes mapping:
* `-v /var/run/docker.sock:/var/run/docker.sock` allows to communicate with host Docker Daemon to run additional containers required in some steps
* `-v <path-to-model-catalog>:<path-to-model-catalog>` the path to the catalog where your model is being stored
* `-v <path-to-workdir>:<path-to-workdir>` the path where Model Navigator commands are executed inside container

**Understanding volumes mounting and workspace**

The commands executed by the Triton Model Navigator creates workspace directory to store the artifacts and share them between the steps.
Additionally, some steps are run inside a separate Docker container maintained by the Triton Model Navigator.

In order to run the Triton Model Navigator inside the Docker container the commands must be executed from **mounted host path**.

Running the Triton Model Navigator container:
```shell
$ docker run -it --rm \
--gpus 1 \
-v /var/run/docker.sock:/var/run/docker.sock \
-v <path-to-model-catalog>:<path-to-model-catalog> \
-v <path-to-workdir>:<path-to-workdir> \
-w <path-to-workdir> \
--net host \
--name model-navigator \
model-navigator /bin/bash

root@hostname:<path-to-workdir>#
```

Additional constraints when using `triton_launch_mode=docker`:
* The mappings of volumes inside the container **must match** the host path name, example: `-v <host-path-name>:<host-path-name>`
* Use `--ipc=host` mode running `model-navigator` container

## Installing from the Source

The Model Navigator can be installed from the source:
```shell
$ make install-with-cli
```
or using Pip:
```shell
pip install --extra-index-url https://pypi.ngc.nvidia.com .[cli]
```
