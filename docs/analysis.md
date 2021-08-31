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

# Models Analysis

The Triton Model Navigator uses the [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) for performing
analysis of profiled models according to provided constraints and objectives.
The analysis step selects the top N model configurations across all prepared versions of models and
applied optimizations.

The top N model configurations detailed report is being stored in `{workspace_path}/analyze_report.pdf`.

## The `analyze` Command

The Triton Model Navigator `analyze` command runs the Triton Model Analyzer to evaluate results stored by `profile` stage.

Using CLI arguments:

```shell
$ model-navigator analyze --workspace-path navigator_workspace \
  --model-repository model-store \
  --max-latency-ms 100 \
  --min-throughput 750
```

Using YAML file:

```yaml
workspace_path: navigator_workspace
model_repository: model-store
max_latency_ms: 100
min_throughput: 750
```

Running command using YAML configuration:

```shell
$ model-navigator analyze --config-path model_navigator.yaml
```

## Constrains

The constraints are the limits in which the analyzed models should match. The default configuration does not set any constraints that models must match; therefore,
the Triton Model Navigator returns all models sorted by the inference throughput.

If a model has to match a maximum latency budget or minimal performance, the flags with values should be passed to the Triton Model Navigator.

The Triton Model Navigator returns top N models matching the given constraints sorted by throughput.

## Objectives

The top N models are sorted by throughput by default; however, the user can provide their own
objectives based on which top N models are presented after the analysis.

The sort order can be changed by defining objectives based on which top N models should be selected and ordered in the
final results:

```yaml
objectives:
    - perf_latency
    - perf_throughput
```

The values can be weighted:

```yaml
objectives:
    perf_latency: 10
    perf_throughput: 5
```

Learn more about Model
Analyzer [objectives here](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md#objective)


## CLI and YAML Config Options

[comment]: <> (START_CONFIG_LIST)
```yaml
# Path to the configuration file containing default parameter values to use. For more information about configuration
# files, refer to: https://github.com/triton-inference-server/model_navigator/blob/main/docs/run.md
[ config_path: path ]

# Path to the output workspace directory.
[ workspace_path: path | default: navigator_workspace ]

# Clean workspace directory before command execution.
[ override_workspace: boolean ]

# NVIDIA framework and Triton container version to use (refer to https://docs.nvidia.com/deeplearning/frameworks/support-
# matrix/index.html and https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html for
# details).
[ container_version: str | default: 21.08 ]

# Custom framework docker image to use. If not provided
# nvcr.io/nvidia/<framework>:<container_version>-<framework_and_python_version> will be used
[ framework_docker_image: str ]

# Custom Triton Inference Server docker image to use. If not provided nvcr.io/nvidia/tritonserver:<container_version>-py3
# will be used
[ triton_docker_image: str ]

# List of GPU UUIDs to be used for the conversion and/or profiling. Use 'all' to profile all the GPUs visible by CUDA.
[ gpus: str | default: ['all'] ]

# Provide verbose logs.
[ verbose: boolean ]

# Path to the Triton Model Repository.
[ model_repository: path | default: model-store ]

# Number of top final configurations selected from the analysis.
[ top_n_configs: integer | default: 3 ]

# The Model Navigator uses the objectives described here to find the best configuration for the model.
[ objectives: list[str] | default: ['perf_throughput=10'] ]

# Maximum latency in ms that the analyzed models should match.
[ max_latency_ms: integer ]

# Minimal throughput that the analyzed models should match.
[ min_throughput: integer | default: 1 ]

# Maximal GPU memory usage in MB that analyzed model should match.
[ max_gpu_usage_mb: integer ]

```
[comment]: <> (END_CONFIG_LIST)
