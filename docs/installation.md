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
# Installation

## Requirements

Ensure your system fulfills the following requirements:
* [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/), [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/), or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

## Using Docker Container

The recommended way of using the Triton Model Navigator is to build a Docker container with all necessary dependencies:

```shell
$ make docker
```

Running the Triton Model Navigator container requires additional mappings:
* `-v /var/run/docker.sock:/var/run/docker.sock` allows running Docker containers as sibling containers from inside the Triton Model Navigator container to perform optimization processes.
* `-v <path-to-model-catalog>:<path-to-model-catalog>` The path to the catalog where models are being stored.
* `-v ${HOME}:${HOME}` The path to the catalog where Model Navigator will be run from. The simple option is to map the user directory. Navigator creates a workspace catalog there to share artifacts across the steps.

**Note**

When using `triton_launch_mode=docker`:
* The mappings of volumes inside the container must match the host path, example: `-v {host-path}:{host-path}`
* Use `--ipc=host` mode running `model-navigator` container


To run the Triton Model Navigator container:
```shell
$ docker run -it \
--gpus 1 \
-v <path-to-model-catalog>:<path-to-model-catalog> \
-v /var/run/docker.sock:/var/run/docker.sock \
-v ${HOME}:${HOME}
-w ${HOME} \
--net host \
--name model-navigator \
model-navigator /bin/bash

root@hostname:/home/{username}#
```

## Installing from the Source

The Model Navigator can be installed from the source:
```shell
$ make install
```

If you are using this approach, you need to install DCGM. In order to install it on Ubuntu 20.04, run:
```shell
$ export DCGM_VERSION=2.0.13
$ wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb
```

The additional packages required:
```shell
$ sudo apt install wkhtmltopdf
```
