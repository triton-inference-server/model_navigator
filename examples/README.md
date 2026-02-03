<!--
Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.

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

1. [Optimize model](./01_optimize_torch_linear_model/)
2. [Optimize model and verify model](./02_optimize_and_verify_model/)
3. [Optimize model and save package](./03_optimize_model_and_save_package/)
4. [Load and optimize package](./04_load%E2%80%8E_and_optimize_package%E2%80%8E/)
5. [Optimize and serve model on PyTriton](./05_optimize_and_serve_model_on_pytriton/)
6. [Optimize and serve model on Triton Inference Server](./06_optimize_and_serve_model_on_triton/)
7. [Optimize model and use for offline inference](./07_optimize_model_and_use_for_offline_inference/)
8. [Optimize PyTorch QAT model](./08_optimize_pytorch_hifigan_qat_model/)
9. [Custom configuration for optimize](./09_custom_configurations_for_optimize/)
10. [Inplace Optimize of single model](./15_inplace_resnet/)
11. [Inplace Optimize of models pipeline](./16_inplace_stable_diffusion/)


## Example models
Inside the [example/models](./models/) directory you can find ready to use example models in various frameworks.

`Python`:
- [Identity Model](./models/python/identity)

`PyTorch`:

- [BART (Inplace Optimize)](./19_inplace_bart)
- [BERT](./models/torch/bert)
- [Linear Model](./models/torch/linear)
- [ResNet50](./models/torch/resnet50_nvidia_deep_learning_examples)
- [Torch Hub ResNet50](./models/torch/resnet50_nvidia_torch_hub)
- [ResNet50 (Inplace Optimize)](./15_inplace_resnet)
- [Stable Diffusion (Inplace Optimize)](./16_inplace_stable_diffusion)
- [Whisper (Inplace Optimize)](./17_inplace_whisper)

`TensorFlow`:

- [Linear Model](./models/tensorflow/linear)
- [EfficientNet](./models/tensorflow/efficientnet)
- [BERT](./models/tensorflow/bert)

`JAX`:

- [Linear Model](./models/jax/linear)
- [GPT-2](./models/jax/gpt2)

`ONNX`:

- [Identity Model](./models/onnx/identity)
