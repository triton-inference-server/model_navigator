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
# Model optimizations

Model Navigator configures the model on Triton Inference Server,
thus it needs to prepare the model files in [formats handled by Triton Inference Server](https://github.com/triton-inference-server/server/blob/master/docs/model_repository.md#model-files).
Currently, the supported model formats are:
- TensorFlow models in SavedModel format
- PyTorch models in TorchScript format (exported either as script or trace)
- ONNX models

Although Triton Inference Server performs [model optimizations](https://github.com/triton-inference-server/server/blob/master/docs/optimization.md), for better performance in inference,
Model Navigator tries to carry out additional optimizations independently.

## Available optimizations

### TensorFlow graph optimization

During this optimization, Model Navigator extracts the subgraph, which is crucial for an inference process, and performs graph optimization on the extracted graph.

### Tensorflow to ONNX conversion

Convert a TensorFlow model to ONNX format with ONNX opset from the `onnx-opsets` list.

Model Navigator uses the [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) for this optimization.

### ONNX to TensorRT conversion

Convert an ONNX model to TensorRT Plan with a set of `target-precisions`
and a profile defined by `min-shapes`, `opt-shapes`, and `max-shapes`.

The available values for `target-precisions` list are `fp16` `fp32`, and `tf32`.
When selecting `tf32`, the result model will have TensorFloat precision or float32 -
depending on the hardware's compute capabilities on which the conversion is performed.
For the source model in `fp16`, precision conversion will fail. By default, `target-precision` is set to `[fp16, tf32]`.

It is crucial to pass profile parameters to [models with a dynamic axis](#models-with-dynamic-axis)
(take a look at this section for details on profile definition).
For models with no dynamic axes, profile definition can be omitted. Model Navigator creates the default profile:
`min=1, opt=max_batch_size, max=max_batch_size`. Where `max-batch-size` is obtained either as
`max(preferred-batch-sizes)` if the parameter is provided or `max-preferred-batch-size`.

Other parameters which can be configured: `max-workspace-size` (default set to 4294967296 - 4GB).
Take a look at [TensorRT documentation](https://developer.nvidia.com/tensorrt) for additional information on TensorRT Plan building.

This conversion should be performed on target hardware.

Model Navigator uses [Polygraphy tool](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy)
for this optimization.

For example:

```shell
# model with dynamic axes
$ model-navigator \
   --model-name EfficientNet \
   --model-path /storage/efficientnet_tf2_fp32.onnx \
   --min-shapes input_1:1,32,32,3 \
   --opt-shapes input_1:16,116,116,3 \
   --max-shapes input_1:32,224,224,3 \
   --target-precisions fp16,tf32 \

# model with no dynamic axes
$ model-navigator \
   --model-name ResNet50 \
   --model-path /storage/rn50_tf1_fp32.onnx \
   --max-preferred-batch-size 32
```

## Definition of model inputs and outputs

The PyTorch TorchScript model format doesn't contain a complete specification of input and outputs - its <name>, <shape_with_batch_dimension>, and <dtype>.
Therefore, the user has to provide the `inputs` and `outputs` configuration parameters for proper model configuration and benchmark.

Each of the input and output parameters have the format: `<name>:<shape_with_batch_dimension>:<dtype>`.
Where:
<name> can be anything
<shape_with_batch_dimension> should contain the batch dimension  Dynamic axes should be marked with -1 value.
<dtype> should be in a format compatible with `numpy.dtype`.

For example, defined inputs and outputs looks similar to:

```shell
$ model-navigator \
   --model-name EfficientNet \
   --model-path /storage/efficientnet_pyt_fp32.pt \
   --inputs image:-1,-1,-1,3:float32 \
   --outputs probs:-1,1000:float32 \
   --max-shapes image:32,224,224,3
```

```shell
$ model-navigator \
   --model-name FastPitch_PyT \
   --model-path /storage/fastpitch_pyt_fp32.pt \
   --inputs INPUT__0:-1,-1:int64 INPUT__1:-1,1:int64 \
   --outputs OUTPUT__0:-1,80,-1:float16 OUTPUT__1:-1,1:int64 OUTPUT__2:-1,-1:float16 OUTPUT__3:-1,-1:float16 \
   --max-shapes INPUT__0:8,128
```
## Models with a dynamic axis

When providing a model which has dynamic axes on Model Navigator input, it is required to provide at least `max-shapes`
to define the shapes of data used during benchmarking.

We recommend providing a complete input profile with `min-shapes`, `opt-shapes`, and `max-shapes` parameters to achieve better performance results.
Users can extract such profiles from a real dataset by calculating its statistics for each input axis.

Each of the shape parameters have the format: `<input_name>:<shape_with_batch_dimension>`.
In case a colon is present in an input name, surround it with question marks.

For example,  a defined profile looks similar to:

```shell
$ model-navigator \
   --model-name EfficientNet \
   --model-path /storage/efficientnet_tf2_fp32.savedmodel \
   --min-shapes input_1:1,32,32,3 \
   --opt-shapes input_1:16,116,116,3 \
   --max-shapes input_1:32,224,224,3
```

## Models with inputs of sequence length

To perform performance analysis on a model with one input describing the sequence's
length in another model input, it is required to limit the range of values passed
to the input with length values.
To do that, use `--range-values` and `--max-shapes` parameters.

```shell
$ export MAX_SEQUENCE_LENGTH=128
$ export DICTIONARY_SIZE=64
$ export BATCH_SIZE=8
$ model-navigator \
   --model-name FastPitch_TF \
   --model-path /storage/fastpitch_pyt_fp32.savedmodel \
   --value-ranges INPUT__0:0,${DICTIONARY_SIZE} INPUT__1:1,${MAX_SEQUENCE_LENGTH} \
   --max-shapes INPUT__0:${BATCH_SIZE},${MAX_SEQUENCE_LENGTH} INPUT__1:${BATCH_SIZE},1
```

## Verification of conversion correctness

Model Navigator verifies the correctness of conversion using the [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/) tool.

It compares the original and converted model's inference results using [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).
Its `rtol` (relative tolerance) and `atol` (absolute tolerance) parameters
can be set using Model Navigator configuration. Both defaults to `1e-5`.
They can be provided on a per-output basis using a dictionary.
In that case, use an empty string ("") as the key to specifying the default tolerance for outputs not explicitly listed.

```shell
--rtol probabilities:1e-4 1e-5 --atol 1e-5
```

```yaml
rtol:
 probabilities: 1e-4
 "": 1e-5
atol: 1e-5
```

Polygraphy also provides diagnostic logs to verify the size of differences and chooses the appropriate outputs if needed.

```
[I]     Comparing Output: 'probabilities' (dtype=float32, shape=(1, 1001)) with 'probabilities' (dtype=float32, shape=(1, 1001))
[I]     Required tolerances: [atol=6.0722e-05] OR [rtol=1e-05, atol=5.9283e-05] OR [rtol=0.0016148, atol=1e-05]
        Runner: onnxrt-runner-N0-04/09/21-07:15:27       | Stats: mean=0.000999, min=1.4181e-05 at (0, 911), max=0.144 at (0, 644)
        Runner: trt-runner-N0-04/09/21-07:15:27          | Stats: mean=0.000999, min=1.4159e-05 at (0, 911), max=0.14394 at (0, 644)
[E]     FAILED | Difference exceeds tolerance (rtol=1e-05, atol=1e-05)
```
