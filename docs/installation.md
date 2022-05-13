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


## Export From Source

Model Navigator Export API is installed with Model Navigator package.
To install Model Navigator Export API without Model Navigator dependencies use commands:

```shell
$ pip install --extra-index-url https://pypi.ngc.nvidia.com git+https://github.com/triton-inference-server/model_navigator.git@v0.3.0#egg=model-navigator[<extras,>] --upgrade
```

Extras:
- pyt - Model Navigator Export API for PyTorch
- tf - Model Navigator Export API for TensorFlow2
- cli - Model Navigator CLI
- huggingface - Model Navigator Export API for HuggingFace

## Optimize for Triton Inference Server

## Using Docker Container
The recommended way of using the Triton Model Navigator is to build a Docker container with all necessary dependencies:
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

If you are using this approach, you need to install DCGM. In order to install it on Ubuntu 20.04, run:
```shell
$ export DCGM_VERSION=2.0.13
$ wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb
```
