<!--
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

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

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Overview](#overview)
- [Model Config sweeping process](#model-config-sweeping-process)
- [Profile Results](#profile-results)
- [The `profile` Command](#the-profile-command)
- [CLI and YAML Config Options](#cli-and-yaml-config-options)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Overview

The `profile` command evaluates a model on the Triton Inference Server to gather statistics for provided search parameters.

The model's profiling is performed using the [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer),
which performs a robust analysis of models on different configurations, gathering the details for performance, latency,
and resources consumption running given models inference on Triton Inference Server.

The profiling sweeps through the model config parameters, generating different model configurations, and evaluates the performance for each.
It is performed for each model configuration already present in Triton model repository.

The sweeping process takes place as long as there is a significant performance increase between the previous and current
configuration.

## Model Config sweeping process

Triton Model Navigator supports automatic and manual [configuration search modes](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md) offered by Triton Model Analyzer.

To use the automatic mode, it is necessary to provide just max boundaries within the search will be performed.

```shell
$ model-navigator profile --workspace-path navigator_workspace \
  --config-search-max-concurrency 2048 \
  --config-search-max-instance-count 3 \
  --config-search-max-preferred-batch-size 512
```

During this mode, Triton Model Analyzer will automatically search over
[instance_group](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups) and
[dynamic_batching](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher) Model Config settings using
range of [concurrent](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.md#request-concurrency) requests.

The manual configuration search can be enabled by providing the lists of parameters:

```shell
$ model-navigator profile --workspace-path navigator_workspace \
  --model-repository my-storage/model-store \
  --config-search-concurrency 512 1024 \
  --config-search-instance-counts gpu=2,4 \
  --config-search-preferred-batch-sizes 128,256 256
```

During this mode, Triton Model Analyzer search over
[max_batch_size](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#maximum-batch-size),
[instance_group](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups),
[dynamic_batching](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher)
and custom backend parameters of Model Config settings with defined parameter values using
defined level of [concurrent](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.md#request-concurrency) requests.

Learn more about the Triton Model Analyzer [Model Config search here](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md).

## Profile Results

The easiest method to analyze and generate reports of profiling process is to use Triton Model Navigator [analyze command](docs/analysis.md).

The Triton Model Analyzer writes the collected measurements to [checkpoint](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/checkpoints.md) files when profiling.
They are saved into the `<workspace_path>/analyzer/checkpoints` directory. To load them you can use:

```python
import json
from model_analyzer.state.analyzer_state import AnalyzerState

with latest_checkpoint_path.open("r") as checkpoint_file:
    state = AnalyzerState.from_dict(json.load(checkpoint_file))

profiling_configs_and_results = state.get("ResultManager.results")  # contain profiling configs, perf_analyzer args and results
server_only_data = state.get("ResultManager.server_only_data")
gpus = state.get("MetricsManager.gpus")
```

## The `profile` Command

Triton Model Navigator evaluates all Triton models configuration stored in `model_repository` directory.

Using CLI arguments:

```shell
$ model-navigator profile --workspace-path navigator_workspace \
  --model-repository my-storage/model-store \
  --config-search-max-concurrency 1024 \
  --config-search-max-instance-count 5 \
  --config-search-max-preferred-batch-size 32
```

Using YAML file:

```yaml
workspace_path: navigator_workspace
model_repository: my-storage/model-store
config_search_max_concurrency: 1024
config_search_max_instance_count: 5
config_search_max_preferred_batch_size: 32
```

Running command using YAML configuration:

```shell
$ model-navigator profile --config-path model_navigator.yaml
```

## CLI and YAML Config Options

[comment]: <> (START_CONFIG_LIST)
```yaml
# Path to the Triton Model Repository.
model_repository: path

# Path to the configuration file containing default parameter values to use.
# For more information about configuration files, refer to:
# https://github.com/triton-inference-server/model_navigator/blob/main/docs/run.md
[ config_path: path ]

# Path to the output workspace directory.
[ workspace_path: path | default: navigator_workspace ]

# Path to the output package.
[ output_package: path ]

# Clean workspace directory before command execution.
[ override_workspace: boolean ]

# NVIDIA framework and Triton container version to use. For details refer to
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html and
# https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html for details).
[ container_version: str | default: 22.02 ]

# Custom framework docker image to use. If not provided
# nvcr.io/nvidia/<framework>:<container_version>-<framework_and_python_version> will be used
[ framework_docker_image: str ]

# Custom Triton Inference Server docker image to use.
# If not provided nvcr.io/nvidia/tritonserver:<container_version>-py3 will be used
[ triton_docker_image: str ]

# List of GPU UUIDs or Device IDs to be used for the conversion and/or profiling.
# All values have to be provided in the same format.
# Use 'all' to profile all the GPUs visible by CUDA.
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
[ perf_measurement_interval: integer | default: 5000 ]

# The method used  to launch the Triton Server.
# 'local' assume tritonserver binary is available locally.
# 'docker' pulls and launches a triton docker container with the specified version.
[ triton_launch_mode: choice(local, docker) | default: local ]

# Path to the Triton Server binary when the local mode is enabled.
[ triton_server_path: str | default: tritonserver ]

# Max concurrency used for automatic config search in analysis.
[ config_search_max_concurrency: integer | default: 1024 ]

# Max number of model instances used for automatic config search in analysis.
[ config_search_max_instance_count: integer | default: 5 ]

# [Deprecated] Maximum preferred batch size allowed for inference used for automatic config search in analysis.
[ config_search_max_preferred_batch_size: integer | default: 32 ]

# List of concurrency values used for manual config search in analysis.
# Forces manual config search.
# Format: --config-search-concurrency 1 2 4 ...
[ config_search_concurrency: list[integer] ]

# List of model instance count values used for manual config search in analysis.
# Forces manual config search.
# Format: --config-search-instance-counts <DeviceKind>=<count>,<count> <DeviceKind>=<count> ...
[ config_search_instance_counts: list[str] ]

# List of max batch sizes used for manual config search in analysis. Forces manual config search.
# Format: --config-search-max-batch-sizes 1 2 4 ...
[ config_search_max_batch_sizes: list[integer] ]

# List of preferred batch sizes used for manual config search in analysis.
# Forces manual config search.
# Format: --config-search-preferred-batch-sizes 4,8,16 8,16 16 ...
[ config_search_preferred_batch_sizes: list[str] ]

# List of custom backend parameters used for manual config search in analysis.
# Forces manual config search.
# Format: --config-search-backend-parameters <param_name1>=<value1>,<value2> <param_name2>=<value3> ...
[ config_search_backend_parameters: list[str] ]

# Map of features names and minimum shapes visible in the dataset.
# Format: --min-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN
[ min_shapes: list[str] ]

# Map of features names and optimal shapes visible in the dataset.
# Used during the definition of the TensorRT optimization profile.
# Format: --opt-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN
[ opt_shapes: list[str] ]

# Map of features names and maximal shapes visible in the dataset.
# Format: --max-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN
[ max_shapes: list[str] ]

# Map of features names and range of values visible in the dataset.
# Format: --value-ranges <input0>=<lower_bound>,<upper_bound> ..
# <inputN>=<lower_bound>,<upper_bound> <default_lower_bound>,<default_upper_bound>
[ value_ranges: list[str] ]

# Map of features names and numpy dtypes visible in the dataset.
# Format: --dtypes <input0>=<dtype> <input1>=<dtype> <default_dtype>
[ dtypes: list[str] ]

```
[comment]: <> (END_CONFIG_LIST)
