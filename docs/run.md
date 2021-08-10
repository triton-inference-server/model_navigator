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

# Running the Triton Model Navigator

The Triton Model Navigator supports a single command to run through the process and step-by-step execution going through each stage.

The `run` command replaces the old default behavior where all the steps are being performed one by one. Review the other commands to learn more about the process:
- [Model Conversion](conversion.md)
- [Triton Model Configurator](triton_model_configurator.md)
- [Profiling](profiling.md)
- [Analysis](analysis.md)
- [Helm Chart Generator](helm_charts.md)

## The `run` Command

The Model Navigator `run` command performs step-by-step execution of model optimization and profiling.
The `run` operations start the model conversion to optimized formats like the TensorRT plan, verify
the conversion correctness, evaluate optimized model versions on Triton, and profile them using
the Triton Model Analyzer. In the final stage, the `run` command analyzes the obtained results
based on provided constraints and objectives, and prepares the Helm Charts deployment for
top N configurations on the Triton Inference Server.

Using CLI arguments:

```shell
$ model-navigator run --model-name add_sub \
    --model-path model_navigator/examples/quick-start/model.pt \
    --inputs INPUT__0:-1,16:float32 INPUT__1:-1,16:float32 \
    --outputs OUTPUT__0:-1,16:float32 OUTPUT__1:-1,16:float32 \
    --max-concurrency 256 \
    --max-latency-ms 50 \
    --verbose
```

Using YAML file:

```yaml
model_name: add_sub
model_path: model_navigator/examples/quick-start/model.pt
inputs:
  INPUT__0:
    name: INPUT__0
    shape:
    - -1
    - 16
    dtype: float32
  INPUT__1:
    name: INPUT__1
    shape:
    - -1
    - 16
    dtype: float32
outputs:
  OUTPUT__0:
    name: OUTPUT__0
    shape:
    - -1
    - 16
    dtype: float32
  OUTPUT__1:
    name: OUTPUT__1
    shape:
    - -1
    - 16
    dtype: float32
max_concurrency: 256
max_latency_ms: 50
verbose: true
```

Running the Triton Model Navigator run command:

```shell
$ model-navigator run --config-path model_navigator.yaml
```

## CLI and YAML Config Options

The Triton Model Navigator can be configured with a [YAML](https://yaml.org/) file or via the command-line interface (CLI).
Every flag supported by the CLI is supported in the configuration file.

The placeholders below are used throughout the configuration:

* `text`: a regular string value
* `integer`: a regular integer value
* `boolean`: a regular boolean value (in the CLI it is the form of a boolean flag)
* `choice(<choices>)`: a string which value should equal to one from the listed
* `path`: a string value pointing to the path
* `list[<type>]`: list of values separated by a string where type is defined as arg

A list of all the configuration options supported by both the CLI and YAML config file are shown below.
Brackets indicate that a parameter is optional. For non-list and non-object parameters, the value is set to the specified default.

The CLI flags corresponding to each of the following options are obtained by converting the `snake_case` options
to `--kebab-case`. For example, `model_name` in the YAML would be `--model-name` in the CLI.

[comment]: <> (START_CONFIG_LIST)
```yaml
# Name of the model.
model_name: str

# Path to the model file.
model_path: path

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

# Format of the model. Should be provided in case it is not possible to obtain format from model filename.
[ model_format: choice(torchscript, tf-savedmodel, onnx, trt) ]

# Version of model used by the Triton Inference Server.
[ model_version: str | default: 1 ]

# Signature of the model inputs.
[ inputs: list[str] ]

# Signature of the model outputs.
[ outputs: list[str] ]

# Target format to generate.
[ target_formats: list[str] | default: ['tf-savedmodel', 'onnx', 'trt', 'torchscript'] ]

# Configure TensorRT builder for precision layer selection.
[ target_precisions: list[choice(fp16, fp32, tf32)] | default: ['fp16', 'tf32'] ]

# Generate an ONNX graph that uses only ops available in a given opset.
[ onnx_opsets: list[integer] | default: [13] ]

# The amount of workspace the ICudaEngine uses.
[ max_workspace_size: integer ]

# Absolute tolerance parameter for output comparison. To specify per-output tolerances, use the format: --atol
# [<out_name>=]<atol>. Example: --atol 1e-5 out0=1e-4 out1=1e-3
[ atol: list[str] | default: ['1e-05'] ]

# Relative tolerance parameter for output comparison. To specify per-output tolerances, use the format: --rtol
# [<out_name>=]<rtol>. Example: --rtol 1e-5 out0=1e-4 out1=1e-3
[ rtol: list[str] | default: ['1e-05'] ]

# Maximum batch size allowed for inference. A max_batch_size value of 0 indicates that batching is not allowed for the
# model
[ max_batch_size: integer | default: 32 ]

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

# Batch sizes that the dynamic batcher should attempt to create. In case --max-queue-delay-us is set and this parameter is
# not, default value will be --max-batch-size.
[ preferred_batch_sizes: list[integer] ]

# Max delay time that the dynamic batcher will wait to form a batch.
[ max_queue_delay_us: integer ]

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

# List of concurrency values used for config search in analysis. Disable search over max_concurrency. Format:
# --concurrency 1 2 4 ... N
[ concurrency: list[integer] ]

# List of model instance count values used for config search in analysis. Disable search over max_instance_count in
# profiling. Format: --instance-counts <DeviceKind>=<count> <DeviceKind>=<count> ...
[ instance_counts: list[str] ]

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

# Perf Analyzer measurement timeout in seconds.
[ perf_analyzer_timeout: integer | default: 600 ]

# Perf Analyzer measurement mode. Available: count_windows, time_windows.
[ perf_measurement_mode: str | default: count_windows ]

# Perf Analyzer count windows number of samples to used for stabilization.
[ perf_measurement_request_count: integer | default: 50 ]

# Perf Analyzer time windows time in [ms] used for stabilization.
[ perf_measurement_interval: integer | default: 10000 ]

# Override conversion container if it already exists.
[ override_conversion_container: boolean ]

```
[comment]: <> (END_CONFIG_LIST)
