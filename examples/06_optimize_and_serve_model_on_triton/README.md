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

# Triton Inference Server Linear model deployment

This example, shows how to optimize a simple linear model and deploy it to Triton Inference Server.

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

**Note**: run this command from the main catalog inside the repository

## Run model optimization

In the next step, the optimize process will be performed for the model.

```bash
python examples/triton/optimize.py
```

Once the process is done, the `model_repository` catalog is created in the current working directory.
At this point, it exits the container.

```bash
exit
```

## Start Triton Inference Server

Based on the created deployment in model repository, the Triton Inference Server can be executed.
The following command starts the server in background mode and exposes the HTTP and gRPC ports.

```bash
docker run --gpus=1 --rm -d \
  --name tritonserver \
  -p8000:8000 \
  -p8001:8001 \
  -p8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.01-py3 \
  tritonserver --model-repository=/models
```

## Use Perf Analyzer to profile the model

Finally, you can run container with Perf Analyzer:
```shell
docker run -it --network=host nvcr.io/nvidia/tritonserver:23.01-py3-sdk bash
```

And profile the model:

```bash
perf_analyzer -m linear --concurrency-range 2:32:2
```

## Remove containers

After finishing running the example, remove the Triton container working in the background:
```
docker stop tritonserver
```
