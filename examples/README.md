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

We provide step-by-step examples that demonstrate how to use various features of Model Navigator.
For the sake of readability and accessibility, we use a simple `torch.nn.Linear` model as an example.
These examples illustrate how to optimize, test and deploy the model on
the [PyTriton](https://github.com/triton-inference-server/pytriton) and [Triton Inference Server](https://github.com/triton-inference-server/server).


## Step-by-step examples

1. [Optimize model](../examples/01_optimize_torch_linear_model/)
2. [Optimize model and verify model](../examples/02_optimize_and_verify_model/)
3. [Optimize model and save package](../examples/03_optimize_model_and_save_package/)
4. [Load and optimize package](../examples/04_load%E2%80%8E_and_optimize_package%E2%80%8E/)
5. [Optimize and serve model on PyTriton](../examples/05_optimize_and_serve_model_on_pytriton/)
6. [Optimize and serve model on Triton Inference Server](../examples/06_optimize_and_serve_model_on_triton/)
7. [Optimize model and use for offline inference](../examples/07_optimize_model_and_use_for_offline_inference/)
8. [Optimize PyTorch QAT model](../examples/08_optimize_pytorch_hifigan_qat_model/)
9. [Custom configuration for optimize](../examples/09_custom_configurations_for_optimize/)
10. [Inplace Optimize of single model](../examples/15_inplace_resnet)
11. [Inplace Optimize of models pipeline](../examples/16_inplace_stable_diffusion)


## Example models
Inside the [example/models](../examples/models/) directory you can find ready to use example models in various frameworks.

`Python`:
- [Identity Model](../examples/models/python/identity)

`PyTorch`:

- [BART (Inplace Optimize)](../examples/19_inplace_bart)
- [BERT](../examples/models/torch/bert)
- [Linear Model](../examples/models/torch/linear)
- [ResNet50](../examples/models/torch/resnet50)
- [ResNet50 (Inplace Optimize)](../examples/15_inplace_resnet)
- [Stable Diffusion (Inplace Optimize)](../examples/16_inplace_stable_diffusion)
- [Whisper (Inplace Optimize)](../examples/18_inplace_whisper)

`TensorFlow`:

- [Linear Model](../examples/models/tensorflow/linear)
- [EfficientNet](../examples/models/tensorflow/efficientnet)
- [BERT](../examples/models/tensorflow/bert)

`JAX`:

- [Linear Model](../examples/models/jax/linear)
- [GPT-2](../examples/models/jax/gpt2)

`ONNX`:

- [Identity Model](../examples/models/onnx/identity)
