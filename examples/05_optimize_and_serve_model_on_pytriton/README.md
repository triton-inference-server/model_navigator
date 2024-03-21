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

# PyTriton Torch linear model deployment

This example, shows how to optimize a simple linear model and deploy it to PyTriton.

## Requirements

The example requires the `torch` package. It can be installed in your current environment using pip:

```shell
pip install torch
```

Or you can use NVIDIA Torch container:
```shell
docker run -it --gpus 1 --shm-size 8gb -v ${PWD}:${PWD} -w ${PWD} nvcr.io/nvidia/pytorch:23.01-py3 bash
```

If you select to use container, we recommend installing
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

## Install the Model Navigator

Install the Triton Model Navigator following the installation guide for Torch:

```shell
pip install --extra-index-url https://pypi.ngc.nvidia.com .[torch]
```

**Note**: run this command from main catalog inside the repository

## Run model optimization

In the next step, the optimize process will be performed for the model.

```bash
python examples/triton/optimize.py
```

Once the process is done, the `linear.nav` package is created in the current working directory.

## Start PyTriton server

This step starts PyTriton server with the package generated in the previous step.

```bash
./serve.py
```

## Example PyTriton client

Use client to test model deployment on PyTriton

```bash
./client.py
```
