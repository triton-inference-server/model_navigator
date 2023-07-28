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

# Optimize and serve ONXX and TensorRT ensemble on PyTriton

In this example we show how optimize ONNX and TensorRT models and build zero-copy ensemble in PyTriton server.

## Requirements

The example requires the `torch` package. It can be installed in your current environment using pip:

```shell
pip install torch
```

Or you can use NVIDIA Torch container:
```shell
docker run -it --gpus 1 --shm-size 8gb -v ${PWD}:${PWD} -w ${PWD} nvcr.io/nvidia/pytorch:23.01-py3 bash
```

If you select to use container we recommend to install
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

## Install the Model Navigator

Install the Triton Model Navigator following the installation guide for Torch:

```shell
pip install --extra-index-url https://pypi.ngc.nvidia.com .[torch]
```

**Note**: run this command from main catalog inside the repository

## Exmaple must be executed from it's directory
```bash
cd examples/14_optimize_and_serve_onnx_and_tensorrt_on_pytriton
```

## Generate TensorRT model
TensorRT plan must be generated on target machine.

```bash
python ./generate_tensorrt_model.py
```

## Run model optimization

In next step the optimize process is going to be performed for the models.

```bash
python ./optimize.py
```

Once the process is done, the `onnx_linear.nav` and `tensorrt_linear.nav` packages are created in current working directory.

## Start PyTriton server

This step starts PyTriton server with package generated in previous step.

```bash
python ./serve.py
```

## Example PyTriton client

Use client to test model deployment on PyTriton

```bash
python ./client.py
```
