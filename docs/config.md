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

# Configuring Model Navigator

The Model Navigator can be configured with a [YAML](https://yaml.org/) file or via the command-line interface (CLI).
Every flag supported by the CLI is supported in the configuration file, but some configurations are only supported using
the config file.

The placeholders below are used throughout the configuration:

* `<boolean>`: a boolean that can take `true` or `false` as value.
* `<string>`: a regular string
* `<comma-delimited-list>`: a list of comma separated items.
* `<int>`: a regular integer value.
* `<list>`: a list of values.

## CLI and YAML Config Options

A list of all the configuration options supported by both the CLI and YAML config file are shown below. Brackets
indicate that a parameter is optional. For non-list and non-object parameters the value is set to the specified default.

The CLI flags corresponding to each of the options below are obtained by converting the `snake_case` options
to `--kebab-case`. For example, `model_name` in the YAML would be `--model-name` in the CLI.

```yaml
- Name of model
model_name: <string>

- Path to the model file/directory
model_path: <string>

# The directory where Model Navigator stores the artifacts and temporary files
[ workspace_path: <string> | default: 'workspace' ]

# Number of top final configurations selected from analysis
[ top_n_configs: <int> | default: 3 ]

# Max concurrency used for config search in analysis
[ max_concurrency: <int> | default: 1024 ]

# Max number of model instances used for config search in analysis
[ max_instance_count: <int> | default: 5 ]

# Max preferred batch size used for config search in analysis
[ max_preferred_batch_size: <int> | default: 32 ]

# Concurrency values to be used for the analysis
[ concurrency: <comma-delimited-string|list> | default: None ]

# Number of model instances values to be used for the analysis
[ instance_counts: <comma-delimited-string|list> | default: None ]

# Max latency constraint in milliseconds
[ max_latency_ms: <int> | default: None ]

# Min throughput constraint in inference/seconds
[ min_throughput: <int> | default: None ]

# Max used GPU memory constraint in MB
[ max_gpu_usage_mb: <int> | default: None ]

# The amount of workspace the `ICudaEngine` uses in the optimization
[ max_workspace_size: <int> | default: None ]

# Configure TensorRT builder for precision layer selection
[ target_precisions: <comma-delimited-string|list> | default: "fp16,tf32" ]

# Generate ONNX graphs that use only ops available in a given opset
[ onnx_opsets: <comma-delimited-string|list> | default: "12,13" ]

# The minimum shapes TensorRT optimization profile(s) supports
[ min_shapes: <list> | default: None ]

# The optimal shapes TensorRT optimization profile(s) supports
[ opt_shapes: <list> | default: None ]

# The maximum shapes TensorRT optimization profile(s) supports. Also, defines shapes of input data used during performance analysis.
[ max_shapes: <list> | default: None ]

# Range of values used during performance analysis defined per input. Also, defines shapes of input data used during performance analysis.
[ value_ranges: <list> | default: None ]

# The model inputs and their shapes
[ inputs: <list> | default: None ]

# The model outputs and their shapes
[ outputs: <list> | default: None ]

# Relative tolerance parameter for output comparison
[ rtol: <list> | default: None ]

# Absolute tolerance parameter for output comparison
[ atol: <list> | default: None ]
-
# How Model Navigator will launch Triton Inference Server. It should be either "docker" or "local".
[ triton_launch_mode: <string> ]

# The full path to the `tritonserver` binary executable
[ triton_server_path: <string> | default: tritonserver ]

# The version of Triton Inference Server container (only `triton_launch_mode: docker`)
[ triton_version: <string> | default: 20.12-py3 ]

# The protocol used to communicate with Triton Inference Server. Only 'http' and 'grpc' are allowed for the values.
[ client_protocol: <string> | default: grpc ]

# Logging level
[ verbose: <bool> | default: false ]

# List of GPU UUIDs to be used for the profiling. Use 'all' to profile all the GPUs visible by CUDA.
[ gpus: <string|comma-delimited-list-string> | default: 'all' ]

# Specify path to config yaml file
[ config_file: <string> ]
```

## YAML Only Options

The following config options are supported only by the YAML config file.

```yaml

# List of preferred batch sizes used in analysis - list of values
[ preferred_batch_sizes: <list<list>> | default: None ]

# List of objectives that the user wants to sort the results by
[ objectives: <objective|list> ]

```

## Example usage

Using CLI:

```shell
$ model-navigator --model-name add_sub \
   --model-path model-navigator/examples/quick-start/model.pt \
   --inputs INPUT__0:-1,16:float32 INPUT__1:-1,16:float32 \
   --outputs OUTPUT__0:-1,16:float32 OUTPUT__1:-1,16:float32 \
   --max-concurrency 256 \
   --max-latency-ms 50 \
   --verbose
```

Using YAML:

```yaml
model_name: add_sub
model_path: model-navigator/examples/quick-start/model.pt
inputs:
 - INPUT__0:16:float32
 - INPUT__1:16:float32
outputs:
 - OUTPUT__0:16:float32
 - OUTPUT__1:16:float32
max_concurrency: 256
max_latency_ms: 50
verbose: true
```

Running Model Navigator:

```
$ model-navigator -f config.yaml
```

## Objectives and Preferred Batch Sizes

Manual settings preferred batch sizes and objectives:

```yaml
objectives:
   - perf_througput
   - perf_latency
prefferred_batch_sizes:
   - [ 16, 32 ]
   - [ 32, 64 ]
```

Weighted objectives:

```yaml
objectives:
   perf_througput: 5
   perf_latency: 10
```
