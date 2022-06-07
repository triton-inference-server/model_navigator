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
# Support Matrix

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Verified Models](#verified-models)
  - [Export from Source](#export-from-source)
  - [Optimize for Triton Inference Server](#optimize-for-triton-inference-server)
- [Third-Party Packages](#third-party-packages)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Verified Models

Below is a list of tested models and third-party dependencies that are supported in the current version.

### Export from Source

We have verified that the Model Navigator export function works correctly for the following models. We're publishing
[scripts](../../tests/functional_framework) that run these exports.

| Source                                 | Model                                                                      |
|----------------------------------------|----------------------------------------------------------------------------|
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [ResNet50 PyT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [EfficientNet PyT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [EfficientNet TF2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Classification/ConvNets) |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) |[BERT TF2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT) |
| [HuggingFace](https://huggingface.co/) |[BERT PyT](https://huggingface.co/docs/transformers/model_doc/bert) |
| [HuggingFace](https://huggingface.co/) |[GPT2 PyT](https://huggingface.co/docs/transformers/model_doc/gpt2) |
| [HuggingFace](https://huggingface.co/) |[GPT2 TF2](https://huggingface.co/docs/transformers/model_doc/gpt2) |
| [HuggingFace](https://huggingface.co/) |[DistilBERT PyT](https://huggingface.co/docs/transformers/model_doc/distilbert) |
| [HuggingFace](https://huggingface.co/) |[DistilBERT TF2](https://huggingface.co/docs/transformers/model_doc/distilbert) |

### Optimize for Triton Inference Server

We have verified that the Model Navigator run command works correctly for the following models. We're publishing a subset of
[scripts](../../tests/functional) with [instructions](../tests/README.md#running-functional-tests) on testing.

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
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) |[Jasper PyT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) |


## Third-Party Packages

A set of component versions are imposed by the used `container_version` parameter value.
By default, we're using the `22.05` container version that contains:

- [Triton Inference Server 2.22.0](https://github.com/triton-inference-server/server)
- [PyTorch 1.12.0a0+8a1a93a](https://github.com/pytorch/pytorch/commit/8a1a93a)
- TensorFlow [2.8.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0) / [1.15.5](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.5)
- [TensorRT TensorRT 8.2.5.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
- [Intel OpenVINO 2021.4](https://github.com/openvinotoolkit/openvino/tree/2021.4)
- [Triton Inference Server 2.22.0](https://github.com/triton-inference-server/server/releases/tag/v2.22.0)
- [ONNX Runtime 1.11.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.11.0)

Refer to [the containers support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
for a detailed summary for each version.
