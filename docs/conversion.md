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
# Model Conversions

The Triton Model Navigator gathers a set of tools for model conversions between formats.
Currently, the supported input and target model formats are:

| Input Model Format    | Target Model Format   |
|-----------------------|-----------------------|
| TensorFlow SavedModel | TensorFlow SavedModel |
| PyTorch TorchScript   | PyTorch TorchScript   |
| ONNX   | ONNX   |
|    | TensorRT   |

The list of target formats is a subset of
[formats handled by Triton Inference Server](https://github.com/triton-inference-server/server/blob/master/docs/model_repository.md#model-files).

The Triton Model Navigator determines the input model format based on its file  extension. In case the model format couldn't be matched with a file extension,
you can use the `model_format` [configuration option](#cli-and-yaml-config-options).

In a single `convert` command call, the Triton Model Navigator will execute all converters that can handle requested parameters.
All result model files are saved inside the `{workspace}/converted` directory. Besides the model files, the Triton Model Navigator
also creates the `{workspace}/convert_results.yaml` result file, containing conversion statuses, configurations, and output models details.
When you define the `output_path` parameter, the first successful model will be saved to a pointed path.

Example usages:

```shell
# model with dynamic axes with
# output TensorRT plan with fp16 conversion will be saved to output_path
$ model-navigator convert \
   --model-name EfficientNet \
   --model-path /storage/efficientnet_tf2_fp32.onnx \
   --output-path efficientnet_tf2_fp16.plan \
   --min-shapes input_1:1,32,32,3 \
   --opt-shapes input_1:16,116,116,3 \
   --max-shapes input_1:32,224,224,3 \
   --target-precisions fp16

# model with all static axes
# model_format is required because of missing suffix in model_path filename
# converts onnx model to trt and saves it in {workspace}/converted directory
$ model-navigator convert \
   --model-name ResNet50 \
   --model-path /storage/rn50_tf1_fp32 \
   --model-format onnx \
   --max-batch-size 32
```

## Available Converters

### Tensorflow SavedModel to ONNX conversion

The Triton Model Navigator uses the [tf2onnx tool](https://github.com/onnx/tensorflow-onnx)
for the TensorFlow SavedModel to the ONNX format conversion.

For models with dynamic axes, you should also provide a `max_shape` configuration option.

Example call:

```shell
$ model-navigator convert \
  --model-name ResNet50 \
  --model-path navigator_workspace/convert/ResNet50/resnet50.savedmodel \
  --target-formats onnx \
  # required because this is model with model inputs containing dynamic axes
  --max-shapes image:128,3,600,800
```

## TorchScript to ONNX conversion

The Triton Model Navigator uses the `torch.onnx` API for the TorchScript to the ONNX format conversion.

You must provide `value_ranges` and `dtypes` parameters for each model input to perform this conversion.
For models with dynamic axes, you should also provide a `max_shape` configuration option.
TorchScript format doesn't contain a model signature; thus, it is required to [define model signature with annotation file or CLI](#definition-of-model-inputs-and-outputs).

Example call:

```shell
$ model-navigator convert \
  --model-name ResNet50_trace \
  --model-path navigator_workspace/convert/pytorch_vision/pytorch_vision/ResNet50/resnet50_ts_trace.pt \
  --target-formats onnx \
  # for TorchScript models it's required to define model signature in case of missing annotation file
  --inputs image__0:-1,3,-1,-1:float32 \
  --outputs output__0:-1,1000:float32 \
  # parameters required by ts2onnx converter
  --value-ranges image__0=0,1 \
  --dtypes image__0=float32 \
  # required because this is model with model inputs containing dynamic axes
  --max-shapes image__0:128,3,600,800
```

Example config file

```yaml
# converter set config
target_formats: ["onnx",]
onnx_opsets: [13]
# model signature
inputs:
    image__0:
      name: image__0
      shape: [-1, 3, -1, -1]
      dtype: float32
outputs:
    output__0:
      name: output__0
      shape: [-1, 1000]
      dtype: float32
# comparator config
atol:
    output__0: 0.01
rtol:
    output__0: 0.1
# dataset profile
max_shapes:
    image__0: [128, 3, 600, 800]
value_ranges:
    image__0: [0., 1.]
dtypes:
    image__0: float32
```

And CLI call using above config file:

```shell
$ model-navigator convert \
  --model-name MobileNetV2_trace \
  --model-path MobileNetV2_trace.pt \
  --config-path MobileNetV2_trace_config.yaml
```


### ONNX to TensorRT conversion

The Triton Model Navigator uses [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy)
for ONNX to TensorRT format conversion.

**Note:** Run this conversion on target hardware.

For models with dynamic axes, it is required to provide an [optimization profile](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles);
however, for models with inputs having all static axes, you can neglect the profile as the Triton Model Navigator generates a default one.
More details, refer to the section on  [TensorRT optimization profiles](#tensorrt-optimization-profiles).

Also, refer to the [TensorRT documentation](https://developer.nvidia.com/tensorrt) for additional information on TensorRT Plan building.

#### TensorRT Optimization Profiles

An [optimization profile](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles)
specifies constraints on dynamic dimensions that the TensorRT auto-tuner should use for optimization.

To define an optimization profile, you should provide minimum, optimal, and maximal shapes for each model input.
For example:

```yaml
min_shapes:
    image__0: [1, 3, 300, 300]
opt_shapes:
    image__0: [64, 3, 320, 320]
max_shapes:
    image__0: [128, 3, 600, 800]
```

```shell
$ model-navigator convert \
  --min-shapes image__0=1,3,300,300 \
  --opt-shapes image__0=64,3,320,320 \
  --max-shapes image__0=128,3,600,800
```

It is required to pass at least one profile to the TensorRT Builder.
If you don't provide an optimization profile for models with inputs containing all static axes,
the Triton Model Navigator creates a default one based on model input signature
and `max_batch_size` parameter:

```shell
$ model-navigator convert \
  --min-shapes input__0=1[,<static_axes>...] \
  --opt-shapes input__0=<max_batch_size>[,<static_axes>...] \
  --max-shapes input__0=<max_batch_size>[,<static_axes>...]
```

The Triton Model Navigator currently supports the definition of a single optimization profile.

## Definition of Model Inputs and Outputs

The PyTorch TorchScript model format doesn't contain a model signature specification (I/O `name`, `shape_with_batch_dimension`, and `dtype`).
To enable the processing of models with no signature, you must provide the `inputs` and `outputs`
configuration options. These can be provided as an annotation file or CLI/configuration file.

The annotation file is a configuration file in yaml format, placed under the same path as model with `.yaml` suffix
(`{model_path}.yaml`). Configuration should contain `inputs` and `outputs` [configuration options](#cli-and-yaml-config-options).

Input and output parameters defined in the CLI have the following format: `<name>:<shape_with_batch_dimension>:<dtype>`.

Inputs/outputs parameters items have the following meanings:
- `name` - the name should be meaningful. For TorchScript models, the name should follow the naming convention `<name>__<index>`.
- `shape_with_batch_dimension` - the shape of I/O, including batch axis dynamic axes, should be marked with a -1 value.
- `dtype` - should be I/O dtype in a format compatible with `numpy.dtype`.

Sample CLI call:

```shell
$ model-navigator convert \
   --model-name FastPitch_PyT \
   --model-path /storage/fastpitch_pyt_fp32.pt \
   --inputs input__0:-1,-1:int64 input__1:-1,1:int64 \
   --outputs output__0:-1,80,-1:float16 output__1:-1,1:int64 output__2:-1,-1:float16 output__3:-1,-1:float16 \
   --max-shapes input__0:8,128
```

The following is a sample configuration/annotation file:

```yaml
inputs:
    input__0:
      name: input__0
      shape: [-1, -1]
      dtype: int64
    input__1:
      name: input__1
      shape: [-1, 1]
      dtype: int64
outputs:
    output__0:
      name: output__0
      shape: [-1, 80, -1]
      dtype: float16
    output__1:
      name: output__1
      shape: [-1, 1]
      dtype: int64
    output__2:
      name: output__2
      shape: [-1, -1]
      dtype: float16
    output__3:
      name: output__2
      shape: [-1, -1]
      dtype: float16
```

## Verification of Conversion Correctness

The Triton Model Navigator verifies the correctness of conversion using randomly generated data and
compares the original and converted model's inference results using
[np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).
Set `rtol` (relative tolerance) and `atol` (absolute tolerance) parameters
to configure the comparator. They can be provided on a per-output basis using a dictionary.
In that case, use an empty string ("") as the key to specifying the default tolerance for outputs not explicitly listed.

```shell
--rtol probabilities:1e-4 1e-5 --atol 1e-5
```

```yaml
rtol:
 probabilities: 1e-4
 "": 1e-5
atol:
 "": 1e-5
```

During conversion, the Triton Model Navigator dumps JSON files containing inputs and outputs used during verification.
They are removed when verbose logs output flag is disabled, and conversion succeeds.

To load numpy arrays from these JSON files, use below sample code:

```python
from polygraphy.json import load_json
from polygraphy.comparator import RunResults

inputs_data = load_json("resnet50_ts_script-ts2onnx_op13-polygraphyonnx2trt_tf32.plan.comparator_inputs.json")
results = RunResults.load("resnet50_ts_script-ts2onnx_op13-polygraphyonnx2trt_tf32.plan.comparator_outputs.json")
for input_data, onnx_output, trt_output in zip(
    inputs_data,
    results["onnxrt-runner-N0-07/05/21-11:13:08"],
    results["trt-runner-N0-07/05/21-11:13:08"]
):
    print(input_data["image__0"])
    print(onnx_output["output__0"])
    print(trt_output["output__0"])
```

In case of failed conversion due to a mismatch of output,
if possible, the tolerance parameters values to pass the conversion verification are determined and dumped to logs.

## CLI and YAML Config Options

[comment]: <> (START_CONFIG_LIST)
```yaml
# Name of the model.
model_name: str

# Path to the model file.
model_path: path

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
[ container_version: str | default: 21.07 ]

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

# Format of the model. Should be provided in case it is not possible to obtain format from model filename.
[ model_format: choice(torchscript, tf-savedmodel, onnx, trt) ]

# Version of model used by the Triton Inference Server.
[ model_version: str | default: 1 ]

# Path to the output file.
[ output_path: path ]

# The method by which to launch conversion. 'local' assume conversion will be run locally. 'docker' build conversion
# Docker and perform operations inside it.
[ launch_mode: choice(local, docker) | default: docker ]

# Override conversion container if it already exists.
[ override_conversion_container: boolean ]

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

```
[comment]: <> (END_CONFIG_LIST)
