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
* Installed [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/), [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

## Using Docker container

The recommended way of using Model Navigator is to build a Docker container with all necessary dependencies:

```shell
$ docker build -f Dockerfile -t model-navigator .
```

Running the Model Navigator container requires additional mappings:
* `-v /var/run/docker.sock:/var/run/docker.sock` allows running Docker containers as sibling containers from inside the Model Navigator container to perform optimization processes.
* `-v <path-to-model-catalog>:<path-to-model-catalog>` The ***absolute*** path to the catalog where models are stored. The mapping inside the container must exactly match the host path.

To run Model Navigator container:
```shell
$ docker run -it \
--gpus 1 \
-v <path-to-model-catalog>:<path-to-model-catalog> \
-v /var/run/docker.sock:/var/run/docker.sock \
-w <path-to-model-catalog> \
--net host \
--name model-navigator \
model-navigator /bin/bash

root@hostname:<path-to-model-catalog>#
```

## Installing from source

Model navigator can be installed from source:
```shell
$ make install
```

If you are using this approach, you need to install DCGM. In order to install it on Ubuntu 20.04, run:
```shell
$ export DCGM_VERSION=2.0.13
$ wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb
```

The additional Python packages required:
```
python3-pdfkit
```
