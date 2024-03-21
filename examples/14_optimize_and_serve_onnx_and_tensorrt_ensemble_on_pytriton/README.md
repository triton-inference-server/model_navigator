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

# Optimize and serve ONNX and TensorRT ensemble on PyTriton

In this example, we show how to optimize ONNX and TensorRT models and build a zero-copy ensemble in PyTriton server.

## Requirements

The example requires `CUDA 11.8` and `torch` package. It can be installed in your current environment using pip:

```shell
pip install torch
```

Or you can use NVIDIA Torch container:
```shell
docker run -it --gpus 1 --shm-size 8gb -v ${PWD}:${PWD} -w ${PWD} nvcr.io/nvidia/pytorch:22.12-py3 bash
```

If you select to use container, we recommend installing
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

## Install the Model Navigator

Install the Triton Model Navigator following the installation guide for Torch:

```shell
pip install --extra-index-url https://pypi.ngc.nvidia.com .[torch]
```

**Note**: run this command from the main catalog inside the repository

## Example must be executed from its directory
```bash
cd examples/14_optimize_and_serve_onnx_and_tensorrt_ensemble_on_pytriton
```

## Generate TensorRT model
TensorRT plan must be generated on the target machine.

```bash
python ./generate_tensorrt_model.py
```

## Run model optimization

In the next step, the optimize process will be performed for the models.

```bash
python ./optimize.py
```

Once the process is done, the `onnx_linear.nav` and `tensorrt_linear.nav` packages are created in current working directory.

## Serving model with NVIDIA PyTriton

Before running the server and client, install the NVIDIA PyTriton:
```shell
pip install nvidia-pytriton
```

Next, start the PyTriton server with the package generated in the previous step.
```bash
python ./serve.py
```

Use client to test model deployment on PyTriton

```bash
python ./client.py
```
