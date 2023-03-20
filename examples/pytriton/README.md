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

# Triton Inference Server MLP model deployment

This example show how to optimize simple mlp model and deploy it to PyTriton and Triton Inference Server.

## Requirements

The example requires the `tensorflow` package. It can be installed in your current environment using pip:

```shell
pip install tensorflow
```

Or you can use NVIDIA TensorFlow container:
```shell
docker run -it --gpus 1 --shm-size 8gb -v ${PWD}:${PWD} -w ${PWD} nvcr.io/nvidia/tensorflow:23.01-tf2-py3 bash
```

If you select to use container we recommend to install
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

## Install the Model Navigator

Install the Triton Model Navigator following the installation guide for tensorflow:

```shell
pip install --extra-index-url https://pypi.ngc.nvidia.com .[tensorflow]
```

**Note**: run this command from main catalog inside the repository

## Run model optimization

In next step the optimize process is going to be performed for the model.

```bash
python examples/triton/optimize.py
```

Once the process is done, the `mlp.nav` package is created in current working directory.

## Start PyTriton server

This step starts PyTriton server with package generated in previous step.

```bash
./serve.py
```

## Example PyTriton client

Use client to test model deployment on PyTriton

```bash
./client.py
```
