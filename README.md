<!--
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

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

# Triton Model Navigator

Welcome to [Triton Model Navigator](https://github.com/triton-inference-server/model_navigator), an inference toolkit designed
for optimizing and deploying Deep Learning models with a focus on NVIDIA GPUs. The Triton Model Navigator streamlines the
process of moving models and pipelines implemented in [PyTorch](https://pytorch.org),
[TensorFlow](https://www.tensorflow.org), and/or [ONNX](https://onnx.ai)
to [TensorRT](https://github.com/NVIDIA/TensorRT).

The Triton Model Navigator automates several critical steps, including model export, conversion, correctness testing, and
profiling. By providing a single entry point for various supported frameworks, users can efficiently search for the best
deployment option using the per-framework optimize function. The resulting optimized models are ready for deployment on
either [PyTriton](https://github.com/triton-inference-server/pytriton)
or [Triton Inference Server](https://github.com/triton-inference-server/server).

## Features at Glance

The distinct capabilities of Triton Model Navigator are summarized in the feature matrix:

| Feature                     | Description                                                                                                                                      |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Ease-of-use                 | Single line of code to run all possible optimization paths directly from your source code                                                        |
| Wide Framework Support      | Compatible with various machine learning frameworks including PyTorch, TensorFlow, and ONNX                                                      |
| Models Optimization         | Enhance the performance of models such as ResNET and BERT for efficient inference deployment                                                     |
| Pipelines Optimization      | Streamline Python code pipelines for models such as Stable Diffusion and Whisper using Inplace Optimization, exclusive to PyTorch                |
| Model Export and Conversion | Automate the process of exporting and converting models between various formats with focus on TensorRT and Torch-TensorRT                        |
| Correctness Testing         | Ensures the converted model produce correct outputs validating against the original model                                                        |
| Performance Profiling       | Profiles models to select the optimal format based on performance metrics such as latency and throughput to optimize target hardware utilization |
| Models Deployment           | Automates models and pipelines deployment on PyTriton and Triton Inference Server through dedicated API                                          |

## Documentation

Learn more about Triton Model Navigator features in [documentation](https://triton-inference-server.github.io/model_navigator).

## Prerequisites

Before proceeding with the installation of Triton Model Navigator, ensure your system meets the following criteria:

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: Version `3.9` or newer
- NVIDIA GPU

You can use NGC Containers for PyTorch and TensorFlow which contain all necessary dependencies:

- [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

## Install

The Triton Model Navigator can be installed from `pypi.org`.

### Installing with PyTorch extras

For installing with PyTorch dependencies, use:

```shell
pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[torch]
```

### Installing with TensorFlow extras

For installing with TensorFlow dependencies, use:

```shell
pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[tensorflow]
```

### Installing with onnxruntime-gpu for CUDA 11

The default CUDA version for ONNXRuntime since 1.19.0 is CUDA 12. To install with CUDA 11 support use following extra index url:
```shell
.. --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ ..
```

## Quick Start

The quick start section provides examples of possible optimization and deployment paths provided in Triton Model Navigator.

### Optimize Stable Diffusion with Inplace

The Inplace Optimize allows seamless optimization of models for deployment, such as converting
them to TensorRT, without requiring any changes to the original Python pipelines.


The below code presents Stable Diffusion pipeline optimization. But first, before you run the example install the required
packages:

```shell
pip install transformers diffusers torch
```

Then, initialize the pipeline and wrap the model components with `nav.Module`::

```python
import model_navigator as nav
from transformers.modeling_outputs import BaseModelOutputWithPooling
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def get_pipeline():
    # Initialize Stable Diffusion pipeline and wrap modules for optimization
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.text_encoder = nav.Module(
        pipe.text_encoder,
        name="clip",
        output_mapping=lambda output: BaseModelOutputWithPooling(**output),
    )
    pipe.unet = nav.Module(
        pipe.unet,
        name="unet",
    )
    pipe.vae.decoder = nav.Module(
        pipe.vae.decoder,
        name="vae",
    )
    return pipe
```

Prepare a simple dataloader:

```python
# Please mind, the first element in tuple need to be a batch size
def get_dataloader():
    return [(1, "a photo of an astronaut riding a horse on mars")]
```

Execute model optimization:

```python
pipe = get_pipeline()
dataloader = get_dataloader()

nav.optimize(pipe, dataloader)
```
Once the pipeline has been optimized, you can load explicit the most performant version of the modules executing:

```python
nav.load_optimized()
```

At this point, you can simply use the original pipeline to generate prediction with optimized models directly in Python:
```python
pipe.to("cuda")

images = pipe(["a photo of an astronaut riding a horse on mars"])
image = images[0][0]

image.save("an_astronaut_riding_a_horse.png")
```

An example of how to serve a Stable Diffusion pipeline through PyTriton can be found [here](https://github.com/triton-inference-server/pytriton/tree/main/examples/huggingface_stable_diffusion).

Please read [Error isolation when running Python script](#error-isolation-when-running-python-script) when you plan
to place code in Python script.


### Optimize ResNET and deploy on Triton

Triton Model Navigator support also optimization path for deployment on Triton. This path is supported for nn.Module,
keras.Model or ONNX files which inputs are tensors.

To optimize ResNet50 model from TorchHub run the following code:

```python
import torch
import model_navigator as nav

# Initialize the model
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).eval()

# Wrap model in nav.Module
resnet50 = nav.Module(resnet50, name="resnet50")

# Optimize Torch model loaded from TorchHub
nav.optimize(resnet50, dataloader=[(1, [torch.randn(1, 3, 256, 256)])])
```

Once optimization is done, creating a model store for deployment on Triton is simple as following code:

```python
import pathlib

# Generate the model store from optimized model
resnet50.triton_model_store(
    model_repository_path=pathlib.Path("model_repository"),
)
```

Please read [Error isolation when running Python script](#error-isolation-when-running-python-script) when you plan
to place code in Python script.

### Profile any model or callable in Python

Triton Model Navigator enhances models and pipelines and provides a uniform method for profiling any Python
function, callable, or model. At present, our support is limited strictly to static batch profiling scenarios.

As an example, we will use a simple function that simply sleeps for 50ms:

```python
import time


def custom_fn(input_):
    # wait 50ms
    time.sleep(0.05)
    return input_
```

Let's provide a dataloader we will use for profiling:

```python
# Tuple of batch size and data sample
dataloader = [(1, ["This is example input"])]
```

Finally, run the profiling of the function with prepared dataloader:

```python
nav.profile(custom_fn, dataloader)
```

## Error isolation when running Python script

**Important**: Please review below section to prevent unexpected issues when running `optimize`.

For better error isolation, some conversions and exports are run in separate child processes using multiprocessing in
the `spawn` mode. This means that everything in a global scope will be run in a child process. You can encounter
unexpected issue when the optimization code is place in Python script and executed as:
```shell
python optimize.py
```
To prevent nested optimization, you have to either put the optimize code in:
```python
if __name__ == "__main__":
    # optimization goes here
```
or
```python
import multiprocessing as mp
if mp.current_process().name == "MainProcess":
    # optimization goes here
```

If none of the above works for you, you can run all optimization in a single process at the cost of error isolation by
setting the following environment variable:
```bash
NAVIGATOR_USE_MULTIPROCESSING=False
```

## GPU and Host memory logging
By default GPU and Host memory usage logs are saved in main `navigator.log` file.

Environment variable `NAVIGATOR_USE_SEPARATE_GPU_MEMORY_LOG_FILE=true` allows to redirect memory use logs to separate `gpu_memory.log` file for better log separation.

## Examples

We offer comprehensive, step-by-step [guides](examples) that showcase the utilization of the Triton Model Navigator’s
diverse features. These guides are designed to elucidate the processes of optimization, profiling, testing, and
deployment of models using [PyTriton](https://github.com/triton-inference-server/pytriton) and [Triton Inference Server](https://github.com/triton-inference-server/server).

## Useful Links

* [Changelog](CHANGELOG.md)
* [Support Matrix](docs/support_matrix.md)
* [Known Issues](docs/known_issues.md)
* [Contributing](CONTRIBUTING.md)
