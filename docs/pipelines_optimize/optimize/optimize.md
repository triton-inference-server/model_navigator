<!--
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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

# Optimize Pipelines

The majority of Generative AI models consist of multiple DL pipelines orchestrated through Python code. The models like
Stable Diffusion or HuggingFace pipelines represent how the modern, complex models are developed and deployed directly
from Python.

## Overview of Inplace Optimize

The Inplace Optimize feature of the Triton Model Navigator offers a PyTorch-specific solution that seamlessly
substitutes `nn.Module` objects in your code with enhanced and optimised models.

The Triton Model Navigator Inplace Optimize provides a smooth way of optimizing the model to TensorRT or Torch-TensorRT under
single coherent API directly in your Python source code.

This process is centered around the `nav.Module` wrapper, which is used to decorate your pipeline models. It initiates the
optimization across the entire pipeline when paired with the appropriate dataloader.

The Triton Model Navigator diligently audits the decorated modules, gathering essential metadata about the inputs and outputs.
It then commences the optimization process, akin to that used for individual model optimization. Ultimately, it replaces
the original model with its optimized version directly within the codebase.

The concept is built around the callable and dataloader:

- `callable` - a Python object, function or callable with 1 or more wrapped models to optimize.
- `dataloader` - a method or class generating input data. The data is utilized to perform export and conversion, as well
  as determine the maximum and minimum shapes of the model inputs and create output samples that are used during
  the optimization process.

## Optimizing Stable Diffusion pipeline

The below code presents Stable Diffusion pipeline optimization. First, initialize pipeline and wrap the model components
with `nav.Module`:

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
def get_dataloader():
    # Please mind, the first element in tuple need to be a batch size
    return [(1, "a photo of an astronaut riding a horse on mars")]
```

Execute model optimization:

```python
pipe = get_pipeline()
dataloader = get_dataloader()

nav.optimize(pipe, dataloader)
```

Review all possible options in the [optimize API](api/optimize.md).

## Optimizing ResNet18 model

The Inplace Optimize can be easily used to optimize a single `nn.Module`. The below example shows how to optimize a
ResNet18 model from TorchHub.

First, initialize model from TorchHub:
```python
import torch

resnet18 = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).to("cuda").eval()
```

Next, define a simple dataloader:
```python
dataloader = [(1, torch.rand(1, 3, 224, 224, device="cuda")) for _ in range(150)]
```

Finally, wrap the model and run optimize:
```python
import model_navigator as nav

resnet18 = nav.Module(resnet18, name="resnet18")

nav.optimize(resnet18, dataloader)
```

## Loading optimized modules

Once the pipeline or model has been optimized, you can load explicit the most performant version of the modules executing:

```python
nav.load_optimized()
```

After executing this method, when the optimized version of module exists, it will be used in your pipeline execution
directly in Python.

## Deploying optimized pipeline or model

Once optimization is done, you can use the pipeline for deployment directly from Python. The example
how to serve Stable Diffusion pipeline through PyTriton can be
found [here](https://github.com/triton-inference-server/pytriton/tree/main/examples/huggingface_stable_diffusion).


## Per module configuration

`nav.optimize` sets its configuration to all pipeline modules that do not have the configuration already specified. So, if you need a different configuration for a given module, just set the `module.optimize_config` property.

```python

pipe = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")

pipe.encoder = nav.Module(pipe.encoder, name="encoder")
pipe.encoder.optimize_config = nav.OptimizeConfig(
    target_formats=(
        nav.Format.TENSORRT,
    ),
    runners=(
        "TensorRT",
    )
)

pipe.decoder = nav.Module(pipe.decoder, name="decoder")
pipe.decoder.optimize_config = nav.OptimizeConfig(
    target_formats=(
        nav.Format.TENSORRT,
        nav.Format.ONNX,
    ),
    runners=(
        "TensorRT",
        "OnnxCUDA", # try also other runner
    )
)

nav.optimize(pipe, dataloader)
```


