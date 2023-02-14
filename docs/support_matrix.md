<!--
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

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

Please find below information about tested models, used environment and libraries.

## Verified Models

We have verified that the NVIDIA Model Navigator Optimize API works correctly for the following models.

| Source                                                                        | Model                                                                                                             |
|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [ResNet50 PyT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets)        |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [EfficientNet PyT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets)    |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [EfficientNet TF2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Classification/ConvNets) |
| [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | [BERT TF2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT)          |
| [HuggingFace](https://huggingface.co/)                                        | [GPT2 Jax](https://huggingface.co/docs/transformers/model_doc/gpt2)                                               |
| [HuggingFace](https://huggingface.co/)                                        | [GPT2 PyT](https://huggingface.co/docs/transformers/model_doc/gpt2)                                               |
| [HuggingFace](https://huggingface.co/)                                        | [GPT2 TF2](https://huggingface.co/docs/transformers/model_doc/gpt2)                                               |
| [HuggingFace](https://huggingface.co/)                                        | [DistilBERT PyT](https://huggingface.co/docs/transformers/model_doc/distilbert)                                   |
| [HuggingFace](https://huggingface.co/)                                        | [DistilGPT2 TF2](https://huggingface.co/docs/transformers/model_doc/gpt2)                                         |

## Third-Party Packages

A set of component versions are imposed by the used NGC container. During testing we have used `23.01` container version
that contains:

- [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
- [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
- [TensorRT 8.5.2.2](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
- [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
- [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.43.1
- [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
- [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)

- Refer to [the containers support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
for a detailed summary for each version.
