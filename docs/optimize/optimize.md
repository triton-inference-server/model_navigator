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

# Optimize Model

As mentioned previously model optimization plays a crucial role in unlocking the maximum performance capabilities of the
underlying hardware. This sections describe in details how Model Navigator perform the optimization to improve the
inference performance and reduce the cost.

## Overview

The Model Navigator optimize process encompasses several crucial steps aimed at improving the performance of deep
learning models and converting them into the most optimal formats. Model Navigator supports various frameworks,
including TensorFlow 2, PyTorch, ONNX, and JAX.

To initiate the multistep conversion and optimization process in `Model Navigator`, users only need to provide the
`model` and `dataloader`. However, for further customization, additional parameters and `custom_configs` can be used to
tailor the optimization process to specific requirements.

The optimization process consists of the following steps:

1. *Model export*: The source deep learning model, created using one of the supported frameworks, is exported to one of
   the intermediaries formats: TorchScript, SavedModel, ONNX.

2. *Model conversion*: The exported model is then converted into a target representation with goal of achieving the best
   possible performance, it includes: TorchTensorRT, TensorFlowTensorRT, ONNX, TensorRT.

3. *Correctness test*: To ensure the correctness of the produced models, Model Navigator performs a series of
   correctness
   tests. These tests calculate absolute and relative tolerance values for source and converted models.

4. *Model profiling*: Model Navigator conducts performance profiling of the converted models. This process
   uses `Navigator Runners` to perform inference and measure its time. The profiler aims to find the maximum throughput
   for each model and calculates its latency. This information can then be used to retrieve the best runners and provide
   you with performance details of the optimal configuration. In that stage a single data sample is used to perform
   profiling.

5. *Verification*: Once the profiling is complete, Model Navigator performs verification tests to validate the metrics
   provided by the user in `verify_func` against all converted models.

## Example Usage

By going through the Optimize process with Model Navigator, deep learning models can be optimized and converted into the
most suitable formats for deployment, with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) often providing the
optimal solution to achieve the best performance.

[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) can be used for applications deployed to the data center, as
well as embedded and automotive environments. It powers key NVIDIA solutions such as NVIDIA TAO, NVIDIA DRIVE™, NVIDIA
Clara™, and NVIDIA Jetpack™.
TensorRT is also integrated with application-specific SDKs, such as NVIDIA DeepStream, NVIDIA Riva, NVIDIA Merlin™,
NVIDIA Maxine™, NVIDIA Morpheus, and NVIDIA Broadcast Engine to provide developers with a unified path to deploy
intelligent video analytics, speech AI, recommender systems, video conference, AI based cybersecurity, and streaming
apps in production.

You can use those default TensorRT compute plans for your deployment to get very good performance for NVIDIA hardware.

You can also apply quantization for some selected models to get better performance like
in [HiFiGAN example](../examples/08_optimize_pytorch_hifigan_qat_model).
This model uses quantization aware
training so accuracy is very good but many other models can use post-training quantization by just enabling INT8 flag in
optimize function. It can reduce accuracy, so you must validate quantized model in such case.

Model Navigator can build for your quantized model, when flag ```INT8``` is used:

```
package = nav.torch.optimize(
    model=model,
    dataloader=dataloader,
    custom_configs=[
            nav.TensorRTConfig(precision=nav.api.config.TensorRTPrecision.INT8),
    ],
)
```

At the end, the summary of the execution is presented and artifacts stored in Navigator workspace, which by default
is ```navigator_workspace``` folder.

## What next?

The result of optimize can be shared in form of [Navigator Package](../package/package.md) or used for inference
deployment on [PyTriton](../pytriton/pytriton_deployment.md)
or [Triton Inference Server](../triton/triton_deployment.md).
