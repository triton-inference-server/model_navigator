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

The recommended way of using the Triton Model Navigator is to build a Docker container with all necessary dependencies:

```shell
$ make docker
```

Run the Triton Model Navigator container from source directory as shown below.
```shell
docker run -it --rm \
 --gpus 1 \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v ${PWD}:${PWD} \
 -w ${PWD} \
 --net host \
 --name model-navigator \
 model-navigator /bin/bash
```

Learn more about installing the Triton Model Navigator using the instructions in the [Installation](installation.md)
section.

## Run the add_sub Example

The [examples/quick-start](../examples/quick-start) directory contains a simple libtorch model that calculates the sum and difference of two inputs.

Run the Triton Model Navigator inside the container from the source directory:
```shell
$ model-navigator run --model-name add_sub \
    --model-path examples/quick-start/model.pt \
    --inputs INPUT__0:-1,16:float32 INPUT__1:-1,16:float32 \
    --outputs OUTPUT__0:-1,16:float32 OUTPUT__1:-1,16:float32
```

Or using configuration stored in YAML file:

```shell
$ model-navigator run --config-path examples/quick-start/model_navigator.yaml
```

***Note:*** Input and output definitions are required for PyTorch models. Read more about that in the [model conversions](conversion.md) section.

You should see an output similar to the output below:
```
2021-12-06 13:23:01 - INFO - model_navigator.log: run args:
2021-12-06 13:23:01 - INFO - model_navigator.log: 	model_name = add_sub
2021-12-06 13:23:01 - INFO - model_navigator.log: 	model_path = examples/quick-start/model.pt
2021-12-06 13:23:01 - INFO - model_navigator.log: 	model_format = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	model_version = 1
2021-12-06 13:23:01 - INFO - model_navigator.log: 	target_formats = ['tf-trt', 'tf-savedmodel', 'onnx', 'trt', 'torchscript']
2021-12-06 13:23:01 - INFO - model_navigator.log: 	onnx_opsets = [13]
2021-12-06 13:23:01 - INFO - model_navigator.log: 	tensorrt_precisions = ['fp16', 'tf32']
2021-12-06 13:23:01 - INFO - model_navigator.log: 	tensorrt_precisions_mode = hierarchy
2021-12-06 13:23:01 - INFO - model_navigator.log: 	tensorrt_explicit_precision = False
2021-12-06 13:23:01 - INFO - model_navigator.log: 	tensorrt_strict_types = False
2021-12-06 13:23:01 - INFO - model_navigator.log: 	tensorrt_sparse_weights = False
2021-12-06 13:23:01 - INFO - model_navigator.log: 	tensorrt_max_workspace_size = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	atol = {'': 1e-05}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	rtol = {'': 1e-05}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	max_batch_size = 1
2021-12-06 13:23:01 - INFO - model_navigator.log: 	inputs = {'INPUT__0': {'name': 'INPUT__0', 'shape': [-1, 16], 'dtype': 'float32'}, 'INPUT__1': {'name': 'INPUT__1', 'shape': [-1, 16], 'dtype': 'float32'}}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	outputs = {'OUTPUT__0': {'name': 'OUTPUT__0', 'shape': [-1, 16], 'dtype': 'float32'}, 'OUTPUT__1': {'name': 'OUTPUT__1', 'shape': [-1, 16], 'dtype': 'float32'}}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	min_shapes = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	opt_shapes = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	max_shapes = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	value_ranges = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	dtypes = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	batching = static
2021-12-06 13:23:01 - INFO - model_navigator.log: 	engine_count_per_device = {}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	triton_backend_parameters = {}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	triton_launch_mode = local
2021-12-06 13:23:01 - INFO - model_navigator.log: 	triton_server_path = tritonserver
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_max_concurrency = 1024
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_max_instance_count = 5
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_max_preferred_batch_size = 32
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_concurrency = []
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_instance_counts = {}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_max_batch_sizes = []
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_preferred_batch_sizes = []
2021-12-06 13:23:01 - INFO - model_navigator.log: 	config_search_backend_parameters = {}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	top_n_configs = 3
2021-12-06 13:23:01 - INFO - model_navigator.log: 	objectives = {'perf_throughput': 10}
2021-12-06 13:23:01 - INFO - model_navigator.log: 	max_latency_ms = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	min_throughput = 1
2021-12-06 13:23:01 - INFO - model_navigator.log: 	max_gpu_usage_mb = None
2021-12-06 13:23:01 - INFO - model_navigator.log: 	perf_analyzer_timeout = 600
2021-12-06 13:23:01 - INFO - model_navigator.log: 	perf_measurement_mode = count_windows
2021-12-06 13:23:01 - INFO - model_navigator.log: 	perf_measurement_request_count = 50
2021-12-06 13:23:01 - INFO - model_navigator.log: 	perf_measurement_interval = 5000
2021-12-06 13:23:01 - INFO - model_navigator.log: 	workspace_path = navigator_workspace
2021-12-06 13:23:01 - INFO - model_navigator.log: 	override_workspace = False
2021-12-06 13:23:01 - INFO - model_navigator.log: 	override_conversion_container = False
2021-12-06 13:23:01 - INFO - model_navigator.log: 	framework_docker_image = nvcr.io/nvidia/pytorch:21.12-py3
2021-12-06 13:23:01 - INFO - model_navigator.log: 	triton_docker_image = nvcr.io/nvidia/tritonserver:21.12-py3
2021-12-06 13:23:01 - INFO - model_navigator.log: 	gpus = ('all',)
2021-12-06 13:23:01 - INFO - model_navigator.log: 	verbose = False
2021-12-06 13:23:03 - INFO - model_navigator.utils.docker: Run docker container with image model_navigator_converter:21.12-py3; using workdir: model_navigator
2021-12-06 13:23:05 - INFO - model_navigator.converter.transformers: Running command copy on examples/quick-start/model.pt
2021-12-06 13:23:05 - INFO - model_navigator.converter.transformers: Running command annotation on model_navigator/navigator_workspace/converted/model.pt
2021-12-06 13:23:05 - INFO - model_navigator.converter.transformers: Saving annotations to model_navigator/navigator_workspace/converted/model.pt.yaml
2021-12-06 13:23:05 - INFO - model_navigator.converter.transformers: Missing model input value ranges required during conversion. Use `value_ranges` config to define missing dataset profiles. Used default values_ranges: {'INPUT__0': (0.0, 1.0), 'INPUT__1': (0.0, 1.0)}
2021-12-06 13:23:05 - INFO - pyt.transformers: ts2onnx command started.
2021-12-06 13:23:08 - INFO - pyt.transformers: ts2onnx command succeed.
2021-12-06 13:23:08 - INFO - polygraphy.transformers: Polygraphy onnx2trt started.
2021-12-06 13:23:08 - WARNING - polygraphy.transformers: This conversion should be done on target GPU platform
2021-12-06 13:23:08 - WARNING - polygraphy.transformers: --max-workspace-size config parameter is missing thus using 4294967296
2021-12-06 13:23:12 - INFO - polygraphy.transformers: Polygraphy onnx2trt succeed.
2021-12-06 13:23:12 - INFO - polygraphy.transformers: Polygraphy onnx2trt started.
2021-12-06 13:23:12 - WARNING - polygraphy.transformers: This conversion should be done on target GPU platform
2021-12-06 13:23:12 - WARNING - polygraphy.transformers: --max-workspace-size config parameter is missing thus using 4294967296
2021-12-06 13:23:15 - INFO - polygraphy.transformers: Polygraphy onnx2trt succeed.
2021-12-06 13:23:17 - INFO - run: Running triton model configuration variants generation for add_sub.ts2onnx_op13
2021-12-06 13:23:17 - INFO - run: Verifying model variant: add_sub.ts2onnx_op13
2021-12-06 13:23:17 - INFO - run: Running triton model configurator for add_sub.ts2onnx_op13

.
.
.
```

The generated models, logs, and Helm Charts can be found in:
```
$PWD
  |--- navigator_workspace
      |--- analyzer
      |--- charts
      |--- converted
      |--- model-store
      |--- logs
      |--- ...
```
