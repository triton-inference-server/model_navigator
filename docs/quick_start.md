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
# Quick Start

The following steps below will guide you through using the Triton Model Navigator to analyze a simple PyTorch model.
The instructions assume a directory structure like the following:

```
$HOME
  |--- model_navigator
      |--- docs
      |--- examples
      |--- model_navigator
      |--- tests
      .
      .
      .
```

## Install the Triton Model Navigator and Run Container

Install the Triton Model Navigator using the instructions in thee [Installation](installation.md)
section, and run the Triton Model Navigator container as shown below.

```shell
docker run -it --rm \
 --gpus 1 \
 -v ${HOME}:${HOME} \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -w ${HOME} \
 --net host \
 --name model-navigator \
 model-navigator /bin/bash
```

## Run the add_sub Example

The [examples/quick-start](../examples/quick-start) directory contains a simple libtorch model that calculates the sum and difference of two inputs.
Run the Triton Model Navigator inside the container with:

```shell
$ model-navigator run --model-name add_sub \
    --model-path model_navigator/examples/quick-start/model.pt \
    --inputs INPUT__0:-1,16:float32 INPUT__1:-1,16:float32 \
    --outputs OUTPUT__0:-1,16:float32 OUTPUT__1:-1,16:float32
```

Or using configuration stored in YAML file:

```shell
$ model-navigator run --config-path model_navigator/examples/quick-start/model_navigator.yaml
```

***Note:*** Input and output definitions are required for PyTorch models. Read more about that in the [model conversions](conversion.md) section.

You should see an output similar to the output below:
```
2021-06-28 08:57:18 - INFO - model_navigator.log: run args:
2021-06-28 08:57:18 - INFO - model_navigator.log: 	model_name = add_sub
2021-06-28 08:57:18 - INFO - model_navigator.log: 	model_path = model_navigator/examples/quick-start/model.pt
2021-06-28 08:57:18 - INFO - model_navigator.log: 	model_format = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	model_version = 1
2021-06-28 08:57:18 - INFO - model_navigator.log: 	target_formats = [<Format.TF_SAVEDMODEL: 'tf-savedmodel'>, <Format.ONNX: 'onnx'>, <Format.TENSORRT: 'trt'>, <Format.TORCHSCRIPT: 'torchscript'>]
2021-06-28 08:57:18 - INFO - model_navigator.log: 	target_precisions = [<TensorRTPrecision.FP16: 'fp16'>, <TensorRTPrecision.TF32: 'tf32'>]
2021-06-28 08:57:18 - INFO - model_navigator.log: 	onnx_opsets = [13]
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_workspace_size = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	atol = {'': 1e-05}
2021-06-28 08:57:18 - INFO - model_navigator.log: 	rtol = {'': 1e-05}
2021-06-28 08:57:18 - INFO - model_navigator.log: 	inputs = {'INPUT__0': {'name': 'INPUT__0', 'shape': (-1, 16), 'dtype': dtype('float32')}, 'INPUT__1': {'name': 'INPUT__1', 'shape': (-1, 16), 'dtype': dtype('float32')}}
2021-06-28 08:57:18 - INFO - model_navigator.log: 	outputs = {'OUTPUT__0': {'name': 'OUTPUT__0', 'shape': (-1, 16), 'dtype': dtype('float32')}, 'OUTPUT__1': {'name': 'OUTPUT__1', 'shape': (-1, 16), 'dtype': dtype('float32')}}
2021-06-28 08:57:18 - INFO - model_navigator.log: 	min_shapes = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	opt_shapes = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_shapes = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	value_ranges = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	dtypes = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_batch_size = 32
2021-06-28 08:57:18 - INFO - model_navigator.log: 	preferred_batch_sizes = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_queue_delay_us = 0
2021-06-28 08:57:18 - INFO - model_navigator.log: 	model_repository = model-store
2021-06-28 08:57:18 - INFO - model_navigator.log: 	triton_launch_mode = TritonLaunchMode.LOCAL
2021-06-28 08:57:18 - INFO - model_navigator.log: 	triton_server_path = tritonserver
2021-06-28 08:57:18 - INFO - model_navigator.log: 	perf_measurement_window = 5000
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_concurrency = 1024
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_instance_count = 5
2021-06-28 08:57:18 - INFO - model_navigator.log: 	concurrency = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	instance_counts = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	top_n_configs = 3
2021-06-28 08:57:18 - INFO - model_navigator.log: 	objectives = {'perf_throughput': 10}
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_latency_ms = 1000
2021-06-28 08:57:18 - INFO - model_navigator.log: 	min_throughput = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	max_gpu_usage_mb = None
2021-06-28 08:57:18 - INFO - model_navigator.log: 	workspace_path = navigator_workspace
2021-06-28 08:57:18 - INFO - model_navigator.log: 	override_workspace = True
2021-06-28 08:57:18 - INFO - model_navigator.log: 	override_container = False
2021-06-28 08:57:18 - INFO - model_navigator.log: 	container_version = 21.05
2021-06-28 08:57:18 - INFO - model_navigator.log: 	gpus = ('all',)
2021-06-28 08:57:18 - INFO - model_navigator.log: 	verbose = False
2021-06-28 08:57:20 - INFO - model_navigator.utils.docker: Run docker container with image model_navigator_converter:21.05-py3; using workdir: /home/{username}
2021-06-28 08:57:22 - INFO - model_navigator.converter.transformers: Running command copy on Projects/JoC/model_navigator/examples/quick-start/model.pt
2021-06-28 08:57:22 - INFO - model_navigator.converter.transformers: Running command annotation on /home/{username}/navigator_workspace/converted/model.pt
2021-06-28 08:57:22 - INFO - model_navigator.converter.transformers: Saving annotations to /home/{username}/navigator_workspace/converted/model.yaml
2021-06-28 08:57:22 - INFO - model_navigator.converter.transformers: Missing model input value ranges required during conversion. Use `value_ranges` config to define missing dataset profile. Used default values_ranges: {'INPUT__0': (0.0, 1.0), 'INPUT__1': (0.0, 1.0)}
2021-06-28 08:57:25 - INFO - pyt.transformers: Optimization succeed.
[I] Loading model: /home/{username}/navigator_workspace/converted/model-ts2onnx_op13.onnx
2021-06-28 08:57:25 - WARNING - polygraphy.transformers: This conversion should be done on target GPU platform
2021-06-28 08:57:25 - WARNING - polygraphy.transformers: --max-workspace-size config parameter is missing thus using 4294967296
2021-06-28 08:57:37 - INFO - polygraphy.transformers: Polygraphy onnx2trt succeed.
[I] Loading model: /home/{username}/navigator_workspace/converted/model-ts2onnx_op13.onnx
2021-06-28 08:57:37 - WARNING - polygraphy.transformers: This conversion should be done on target GPU platform
2021-06-28 08:57:37 - WARNING - polygraphy.transformers: --max-workspace-size config parameter is missing thus using 4294967296
2021-06-28 08:57:49 - INFO - polygraphy.transformers: Polygraphy onnx2trt succeed.
2021-06-28 08:57:50 - INFO - run: Running triton model configuration varians generation for add_sub
2021-06-28 08:57:50 - INFO - run: Running triton model configurator for add_sub
2021-06-28 08:57:51 - INFO - run: Running triton model evaluator for add_sub
.
.
.
```

The generated models, logs, and Helm Charts can be found in:
```
$PWD
  |--- workspace
      |--- analyzer
      |--- charts
      |--- converted
      |--- model-store
      |--- logs
      |--- ...
```
