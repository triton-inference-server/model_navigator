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
# Models Profiling

The `profile` command evaluates a model on the Triton Inference Server in order to gather statistics for provided search parameters.

The model's profiling is performed using the [Model Analyzer](https://github.com/triton-inference-server/model_analyzer),
which performs a robust analysis of models on different configurations, gathering the details for performance, latency,
and resources consumption running given models inference on Triton Inference Server.

The profiling sweeps through the [optimization settings](https://github.com/triton-inference-server/server/blob/master/docs/optimization.md#optimization-settings)
and concurrency, generates different model configurations, and evaluates the performance for each.

The sweep process takes place as long as there is a significant performance increase between the previous and current
configuration. Learn more about the Model Analyzer [config search here](https://github.com/triton-inference-server/model_analyzer/blob/r21.05/docs/config_search.md).

## The `profile` Command

Triton Model Navigator evaluates Triton models configuration stored in `workspace`/`model_repository` directory.

Using CLI arguments:

```shell
$ model-navigator profile --workspace-path navigator_workspace \
  --model-repository model-store \
  --max-concurrency 1024 \
  --max-instance-count 5 \
  --max-batch-size 32
```

Using YAML file:

```yaml
workspace_path: navigator_workspace
model_repository: model-store
max_concurrency: 1024
max_instance_count: 5
max_batch_size: 32
```

Running command using YAML configuration:

```shell
$ model-navigator profile --config-path model_navigator.yaml
```

## Model Config Search

Triton Model Navigator supports automatic and manual configuration search modes offered by [Model Analyzer](https://github.com/triton-inference-server/model_analyzer/blob/r21.05/docs/config_search.md).

In order to use the automatic mode, it is necessary to provide max boundaries within the search will be performed.

```shell
$ model-navigator profile --workspace-path navigator_workspace \
  --max-concurrency 2048 \
  --max-instance-count 3 \
  --max-batch-size 512
```

The manual configuration search can be enabled by providing the lists of parameters:

```shell
$ model-navigator profile --workspace-path navigator_workspace \
  --concurrency 512 1024 \
  --instance-counts gpu=2,4 \
  --preferred-batch-sizes 128 256
```

## CLI and YAML Config Options

[comment]: <> (START_CONFIG_LIST)
```yaml
# Path to the configuration file containing default parameter values to use. For more information about configuration
# files, refer to: https://github.com/triton-inference-server/model_navigator/blob/main/docs/run.md
[ config_path: path | default: model_navigator.yaml ]

# Path to the output workspace directory.
[ workspace_path: path | default: navigator_workspace ]

# Clean workspace directory before command execution.
[ override_workspace: boolean ]

# NVIDIA framework and Triton container version to use (refer to https://docs.nvidia.com/deeplearning/frameworks/support-
# matrix/index.html and https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html for
# details).
[ container_version: str | default: 21.06 ]

# Custom framework docker image to use. If not provided
# nvcr.io/nvidia/<framework>:<container_version>-<framework_and_python_version> will be used
[ framework_docker_image: path ]

# Custom Triton Inference Server docker image to use. If not provided nvcr.io/nvidia/tritonserver:<container_version>-py3
# will be used
[ triton_docker_image: path ]

# List of GPU UUIDs to be used for the conversion and/or profiling. Use 'all' to profile all the GPUs visible by CUDA.
[ gpus: str | default: ['all'] ]

# Provide verbose logs.
[ verbose: boolean ]

# Perf Analyzer measurement timeout in seconds.
[ perf_analyzer_timeout: integer | default: 600 ]

# Perf Analyzer measurement mode. Available: count_windows, time_windows.
[ perf_measurement_mode: str | default: count_windows ]

# Perf Analyzer count windows number of samples to used for stabilization.
[ perf_measurement_request_count: integer | default: 50 ]

# Perf Analyzer time windows time in [ms] used for stabilization.
[ perf_measurement_interval: integer | default: 10000 ]

# Path to the Triton Model Repository.
[ model_repository: path | default: model-store ]

# The method used  to launch the Triton Server. 'local' assume tritonserver binary is available locally. 'docker' pulls
# and launches a triton docker container with the specified version.
[ triton_launch_mode: choice(local, docker) | default: local ]

# Path to the Triton Server binary when the local mode is enabled.
[ triton_server_path: str | default: tritonserver ]

# Max concurrency used for config search in analysis.
[ max_concurrency: integer | default: 1024 ]

# Max number of model instances used for config search in analysis.
[ max_instance_count: integer | default: 5 ]

# Maximum batch size allowed for inference. A max_batch_size value of 0 indicates that batching is not allowed for the
# model
[ max_batch_size: integer | default: 32 ]

# List of concurrency values used for config search in analysis. Disable search over max_concurrency. Format:
# --concurrency 1 2 4 ... N
[ concurrency: list[integer] ]

# List of model instance count values used for config search in analysis. Disable search over max_instance_count in
# profiling. Format: --instance-counts <DeviceKind>=<count> <DeviceKind>=<count> ...
[ instance_counts: list[str] ]

# Batch sizes that the dynamic batcher should attempt to create. In case --max-queue-delay-us is set and this parameter is
# not, default value will be --max-batch-size.
[ preferred_batch_sizes: list[integer] ]

# Map of features names and minimum shapes visible in the dataset. Format: --min-shapes <input0>=D0,D1,..,DN ..
# <inputN>=D0,D1,..,DN
[ min_shapes: list[str] ]

# Map of features names and optimal shapes visible in the dataset. Used during the definition of the TensorRT optimization
# profile. Format: --opt-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN
[ opt_shapes: list[str] ]

# Map of features names and maximal shapes visible in the dataset. Format: --max-shapes <input0>=D0,D1,..,DN ..
# <inputN>=D0,D1,..,DN
[ max_shapes: list[str] ]

# Map of features names and range of values visible in the dataset. Format: --value-ranges
# <input0>=<lower_bound>,<upper_bound> .. <inputN>=<lower_bound>,<upper_bound> <default_lower_bound>,<default_upper_bound>
[ value_ranges: list[str] ]

# Map of features names and numpy dtypes visible in the dataset. Format: --dtypes <input0>=<dtype> <input1>=<dtype>
# <default_dtype>
[ dtypes: list[str] ]

```
[comment]: <> (END_CONFIG_LIST)
