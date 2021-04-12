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

The steps below will guide you through using Model Navigator to analyze a simple PyTorch model. The instructions below assume a directory structure like the
following:

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

## Install Model Navigator and Run Container

Install Model Navigator by following the instructions in
the [Installation](installation.md)
section, and run the Triton Model Navigator container as shown below.

```shell
docker run -it \
 --gpus 1 \
 -v ${HOME}:${HOME} \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -w ${HOME} \
 --net host \
 --name model-navigator \
 model-navigator /bin/bash
```

## Run the add_sub example

The [examples/quick-start](../examples/quick-start) directory contains a simple libtorch model which calculates the sum and difference of two inputs. Run the Model Navigator inside the container with:

```shell
$ model-navigator --model-name add_sub \
    --model-path model_navigator/examples/quick-start/model.pt \
    --inputs INPUT__0:-1,16:float32 INPUT__1:-1,16:float32 \
    --outputs OUTPUT__0:-1,16:float32 OUTPUT__1:-1,16:float32
```

You should see an output similar to the output below:
```
2021-04-12 11:36:49,634 - INFO - model_navigator.log: Model Navigator config:
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	model_name = add_sub
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	model_path = model-navigator/examples/quick-start/model.pt
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	config_file = None
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	workspace_path = workspace
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	verbose = False
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	top_n_configs = 3
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	max_concurrency = 1024
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	max_instance_count = 5
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	max_preferred_batch_size = 32
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	concurrency = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	instance_counts = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	preferred_batch_sizes = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	max_latency_ms = None
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	min_throughput = None
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	max_gpu_usage_mb = None
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	objectives = {'perf_throughput': 10}
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	triton_version = 21.03-py3
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	triton_launch_mode = local
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	triton_server_path = tritonserver
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	client_protocol = grpc
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	gpus = ['all']
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	target_format = None
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	max_workspace_size = None
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	target_precisions = [<Precision.FP16: 'fp16'>, <Precision.TF32: 'tf32'>]
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	onnx_opsets = [12, 13]
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	min_shapes = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	opt_shapes = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	max_shapes = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	value_ranges = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	inputs = [TensorSpec(name='INPUT__0', shape=(-1, 16), dtype=dtype('float32')), TensorSpec(name='INPUT__1', shape=(-1, 16), dtype=dtype('float32'))]
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	outputs = [TensorSpec(name='OUTPUT__0', shape=(-1, 16), dtype=dtype('float32')), TensorSpec(name='OUTPUT__1', shape=(-1, 16), dtype=dtype('float32'))]
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	rtol = []
2021-04-12 11:36:49,634 - INFO - model_navigator.log: 	atol = []
2021-04-12 11:36:49,645 - INFO - model_navigator.entrypoint: Starting Model Navigator
2021-04-12 11:36:49,689 - INFO - numba.cuda.cudadrv.driver: init
2021-04-12 11:36:50,797 - INFO - model_navigator.optimizer: Building optimizer docker image
2021-04-12 11:38:41,857 - INFO - model_navigator.optimizer: Run optimizer
2021-04-12 11:38:42,926 - INFO - model_navigator.optimizer.transformers: Running optimization ts2onnx_op12
2021-04-12 11:38:43,596 - WARNING - pyt.transformers: Optimization failed. Details can be found in logfile: workspace/optimized/model.ts2onnx_op12.onnx.log
2021-04-12 11:38:43,597 - INFO - model_navigator.optimizer.transformers: Running optimization ts2onnx_op13
2021-04-12 11:38:44,108 - WARNING - pyt.transformers: Optimization failed. Details can be found in logfile: workspace/optimized/model.ts2onnx_op13.onnx.log
2021-04-12 11:38:44,609 - INFO - model_navigator.model_navigator: New model: add_sub @ workspace/optimized/model.pt
2021-04-12 11:38:44,609 - INFO - model_navigator.model_navigator: Number of models after optimization: 1
2021-04-12 11:38:44,611 - INFO - model_navigator.model_navigator: Prepared 1 model variants.
2021-04-12 11:38:44,612 - INFO - model_navigator.model_navigator: Verifying variant 1
2021-04-12 11:38:44,621 - INFO - model_navigator.triton.server.server_local: Triton Server started.
2021-04-12 11:38:44,622 - INFO - model_navigator.triton.client: Connecting to grpc://localhost:8001
2021-04-12 11:38:47,649 - INFO - model_navigator.triton.model_store: Deploying model workspace/optimized/model.pt in Triton Model Store workspace/model-store with config ModelConfig(model_name='add_sub.ts-script_none_0', model_version='1', model_format=<Format.TS_SCRIPT: 'ts-script'>, max_batch_size=32, precision=<Precision.ANY: 'any'>, gpu_engine_count=1, preferred_batch_sizes=[16, 32], max_queue_delay_us=1, capture_cuda_graph=0, accelerator=<Accelerator.NONE: 'none'>, inputs=[TensorSpec(name='INPUT__0', shape=(-1, 16), dtype=dtype('float32')), TensorSpec(name='INPUT__1', shape=(-1, 16), dtype=dtype('float32'))], outputs=[TensorSpec(name='OUTPUT__0', shape=(-1, 16), dtype=dtype('float32')), TensorSpec(name='OUTPUT__1', shape=(-1, 16), dtype=dtype('float32'))])
2021-04-12 11:38:51,017 - INFO - model_navigator.deployer.deployer: Evaluating model add_sub.ts-script_none_0 on Triton
2021-04-12 11:39:29,263 - INFO - model_navigator.triton.server.server_local: Triton Server stopped.
2021-04-12 11:39:29,263 - INFO - model_navigator.deployer.deployer: Done. Model add_sub.ts-script_none_0 ready to promote to analysis.
2021-04-12 11:39:29,264 - INFO - model_navigator.model_navigator: Deployment for add_sub.ts-script_none_0 variant succeed. Promoting to analysis stage.
2021-04-12 11:39:29,270 - INFO - model_navigator.analyzer: Copying files from workspace/model-store/add_sub.ts-script_none_0 to workspace/analyzer/model-store/add_sub.ts-script_none_0
2021-04-12 11:39:29,270 - INFO - model_navigator.analyzer: Prepare analysis for 1 models:
2021-04-12 11:39:29,270 - INFO - model_navigator.analyzer: add_sub.ts-script_none_0
.
.
.
```

The generated models, logs and Helm Charts can be found in:
```
$HOME
  |--- workspace
      |--- charts
```
