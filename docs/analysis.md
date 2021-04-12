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

# Model analysis

Model Navigator evaluates a model on Triton Inference Server in order to find the most optimal version within the given
constraints.

The analysis of the model and its optimized version is performed
using the [Model Analyzer](https://github.com/triton-inference-server/model_analyzer). Model Analyzer performs a robust
analysis of models on different configurations and chooses a configuration that maximizes the performance of Triton
Inference Server.

The preliminary step of analysis prepares an additional set of model variants
by applying [framework-specific optimizations](https://github.com/triton-inference-server/server/blob/master/docs/optimization.md#framework-specific-optimization)
available in Triton Inference Server.

## Framework-Specific optimizations

Triton Inference Server provides an additional layer of optimizations for backends used for running models. Model
Navigator
creates a [model configuration](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md)
for Triton Inference Server by
applying [framework-specific optimizations](https://github.com/triton-inference-server/server/blob/master/docs/optimization.md#framework-specific-optimization)
for each model variant prepared in the optimization step.

Before the model and Triton Inference Server configuration can be promoted to the analysis stage, the Model Navigator performs a simple check in order to
verify if a given variant can be run on Triton Inference Server without an issue. In the case of any problems, you can verify the log and help
address problems accordingly.

## Analysis

Model Navigator uses the [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) for the best configuration in order to maximize performance.
The analyzer sweeps through the
[optimization settings](https://github.com/triton-inference-server/server/blob/master/docs/optimization.md#optimization-settings)
and concurrency, generates different model configurations, and evaluates the performance for each.

The sweep process takes place as long as there is a significant performance increase between the previous and current
configuration. Learn more about the Model
Analyzer [config search here](https://github.com/triton-inference-server/model_analyzer/blob/r21.03/docs/config_search.md).

## Config search

Model Navigator provides the following default values for the config search settings:

```yaml
max_concurrency: 1024
max_instance_count: 5
max_preferred_batch_size: 32
```

These values should be adjusted per model and can be overwritten in the Model Navigator configuration:

```shell
$ model-navigator \
   --model-name resnet50 \
   --model-path /storage/resnet50.savedmodel \
   --max-concurrency 2048 \
   --max-instance-count 3 \
   --max-preferred-batch-size 256
```

The search process starts from value 1 of each parameter and evaluates the configs until it reaches maximum defined values or there
is no significant performance increase between the previous and current configuration.

If you want to shorten the search process, or you are aware of a potential best configuration, you might provide
search values manually in the yaml file:

```yaml
model_name: resnet50
model_path: /storage/resnet50.savedmodel
concurrency: [ 32, 64, 128 ]
instance_counts: [ 2, 3, 4 ]
preferred_batch_sizes:
   - [ 16 32 ]
   - [ 32 64 ]
```

Run Model Navigator passing config file as an argument:

```shell
$ model-navigator -f config.yaml
```

## Constrains

The constraints are the limits in which the analyzed models should match. The default configuration does not set any
constraints that models must match, therefore, the Model Navigator returns all models sorted by the inference throughput.

If a model has to match a maximum latency budget or minimal performance, the flags with values should be passed to Model
Navigator.

```shell
$ model-navigator \
   --model-name resnet50 \
   --model-path /storage/resnet50.savedmodel \
   --max-latency-ms 100 \
   --min-throughput 750
```

Model Navigator will return top N models matching the given constraints sorted by throughput.

## Objectives

The top N models are sorted by throughput by default, however the user can provide their own objectives based on which top N
models are presented after the analysis.

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
