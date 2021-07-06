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
# Support Matrix

Below is a list of tested models and third-party dependencies that are supported in the current version.

## Verified Models

We have verified that the Model Navigator run command works correctly for the following models. We're publishing a subset of
[scripts](../tests/functional) with [instructions](../tests/README.md#running-functional-tests) on testing.

Refer to the [changelog](../CHANGELOG.md) for any related issues on these models.

| Source                                 | Model                                                                      |
|----------------------------------------|----------------------------------------------------------------------------|
| [PyTorch Torchvision](https://pytorch.org/vision/master/models.html) | [ResNet50](https://pytorch.org/hub/pytorch_vision_resnet/) (scripted + traced) |
| [PyTorch Torchvision](https://pytorch.org/vision/master/models.html) | [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) (scripted and traced) |
| [PyTorch Torchvision](https://pytorch.org/vision/master/models.html) | [InceptionV3](https://pytorch.org/hub/pytorch_vision_inception_v3/) (traced)        |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [ResNet50 TF1](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets) |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [ResNet50 PyT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [EfficientNet TF2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Classification/ConvNets) |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) |[FastPitch PyT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch) |

## Third-Party Packages

A set of component versions are imposed by the used `container_version` parameter value.
By default, we're using the 21.05 container version that contains:

- PyTorch 1.9.0a0+2ecb2c7
- TensorFlow 2.4.0 / 1.15.5
- TensorRT 7.2.3.4
- [Triton Inference Server 2.10.0](https://github.com/triton-inference-server/server/releases/tag/v2.10.0)
  - [ONNX Runtime 1.7.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.7.0) (support for opset 13 from with ONNX 1.8)
- [Polygraphy 0.29.1](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy)

Refer to [the containers support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
for a detailed summary for each version.

The Triton Model Navigator installs additional components:

- [tf2onnx 1.8.5](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.8.5)
- [Triton Model Analyzer 21.05](https://github.com/triton-inference-server/model_analyzer)
