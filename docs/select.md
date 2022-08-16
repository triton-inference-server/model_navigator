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

# Select the best model configuration and produce a model repository for the Triton Inference Server

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Synopsis](#synopsis)
- [Description](#description)
- [Examples](#examples)
  - [Basic usage](#basic-usage)
  - [Specifying additional objectives and constraints](#specifying-additional-objectives-and-constraints)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Synopsis

```shell
$ model-navigator select [<options>] [ _triton-package_ ]
```

## Description

The `model-navigator select` command selects the best model configuration from input package,
according to user-specified objectives and constraints. It outputs a model repository that can be
directly deployed with the Triton Inference Server.

## Examples

### Basic usage

```
$ model-navigator select my_model.triton.nav
```

### Specifying additional objectives and constraints

```
$ model-navigator select my_model.triton.nav \
                        --objective perf_throughput=10 perf_latency_avg=5  # objectives with weights
                        --max-latency-ms 1  \
                        --min-throughput 100  \
                        --max-gpu-usage-mb 8000  \
                        --target-format trt onnx
```


[comment]: <> (START_CONFIG_LIST)
```yaml
[ output_path: path | default: model_repository ]

# Overwrite any existing model repository at the output path.
[ override: boolean ]

# Provide verbose logs.
[ verbose: boolean ]

# Pick a particular model configuration. If specified, other selection options are ignored.
[ model_config_name: str ]

# Objective used to rank those configurations of the model that fulfill other constraints, with an optional weight value.
# Can be passed multiple times to specify multiple objectives. Available objectives: 'perf_throughput',
# 'perf_latency_p99'.
[ objective: list[str] | default: ['perf_throughput=10'] ]

# Maximum latency in ms that the analyzed models should match.
[ max_latency_ms: integer ]

# Minimal throughput that the analyzed models should match.
[ min_throughput: integer ]

# Maximal GPU memory usage in MB that analyzed model should match.
[ max_gpu_usage_mb: integer ]

# Select only from among models in the given format(s). Multiple formats can be provided as a space-separated list, or by
# passing the option multiple times.
[ target_format: list[str] | default: ['tf-trt', 'tf-savedmodel', 'onnx', 'trt', 'torchscript', 'torch-trt'] ]

# Select only from among model configurations using given backend accelerator.
[ backend_accelerator: list[choice(none, amp, trt, openvino)] ]

# Select only from among model configurations using given precision for TensorRT acceleration.
[ tensorrt_precision: list[choice(int8, fp16, fp32)] ]

# Select only from among model configurations using the CUDA capture graph feature on the TensorRT backend.
[ tensorrt_capture_cuda_graph: boolean ]

```
[comment]: <> (END_CONFIG_LIST)
