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

# Examples

We provide examples how to use Model Navigator to optimize models in frameworks (PyTorch, TensorFlow2, JAX, ONNX), from
existing .nav packages, and also how to deploy optimized models on
the [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).

## Optimize models in frameworks

You can find examples per each supported framework.

`Python`:
- [Identity Model](../examples/python/identity)

`PyTorch`:

- [Linear Model](../examples/torch/linear)
- [ResNet50](../examples/torch/resnet50)
- [BERT](../examples/torch/bert)

`TensorFlow`:

- [Linear Model](../examples/tensorflow/linear)
- [EfficientNet](../examples/tensorflow/efficientnet)
- [BERT](../examples/tensorflow/bert)

`JAX`:

- [Linear Model](../examples/jax/linear)
- [GPT-2](../examples/jax/gpt2)

`ONNX`:

- [Identity Model](../examples/onnx/identity)

## Optimize Navigator Package

The Navigator Package can be reused for optimize e.g. on the new hardware or with newer libraries.
The example code can be found in [examples/package](../examples/package).

### Using model on Triton Inference Server

The optimized model by Triton Model Navigator can be used for serving inference through Triton Inference Server. The
example code can be found in [examples/triton](../examples/triton).
