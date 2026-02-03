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

# Quick Start

These sections provide an overview of optimizing a model, optimizing a pipeline, deploying a model for serving inference
on [PyTriton](https://github.com/triton-inference-server/pytriton)
or the [Triton Inference Server](https://github.com/triton-inference-server/server) as well as using the Navigator Package.
In each section, you will find links to learn more about the Triton Model Navigator
features.

## Optimize Pipeline

The Inplace Optimize feature of the Triton Model Navigator offers a PyTorch-specific solution that seamlessly
substitutes `nn.Module` objects in your code with their enhanced models.

This process is centered around the `nav.Module` wrapper, which is used to decorate your pipeline models. It initiates the
optimization across the entire pipeline when paired with the appropriate dataloader.

The Triton Model Navigator diligently audits the decorated modules, gathering essential metadata about the inputs and outputs.
It then commences the optimization process, akin to that used for individual model optimization. Ultimately, it replaces
the original model with its optimized version directly within the codebase.

The concept is built around the pipeline and dataloader:

- `pipeline` - a Python object or callable with 1 or more wrapped models to optimize.
- `dataloader` - a method or class generating input data. The data is utilized to perform export and conversion, as well
   as determine the maximum and minimum shapes of the model inputs and create output samples that are used during
   the optimization process.

The below code presents Stable Diffusion pipeline optimization. But first, before you run the example install the required
packages:

```shell
pip install transformers diffusers torch
```

Then, initialize the pipeline and wrap the model components with `nav.Module`:

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
        output_mapping=lambda output: BaseModelOutputWithPooling(**output), # Mapping to convert output data to HuggingFace class
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

## Optimize Model

Optimizing models using the Triton Model Navigator is as simple as calling the `optimize` function. The optimization process requires
at least:

- `model` - a Python object, callable, or file path with a model to optimize.
- `dataloader` - a method or class generating input data. The data is utilized to perform export and conversion, as well
   as determine the maximum and minimum shapes of the model inputs and create output samples that are used during
   the optimization process.

Besides the model optimization, the Triton Model Navigator collects information about model shapes, their min and max ranges, validates
the correctness of optimized formats, and improves hardware utilization through searching for maximal throughput on current
hardware.

Here is an example of running `optimize` on the Torch Hub ResNet50 model:

```python
import torch
import model_navigator as nav

# run optimization in the parent process only
package = nav.torch.optimize(
    model=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).eval(),
    dataloader=[torch.randn(1, 3, 256, 256) for _ in range(10)],
)
```

Once the model has been optimized, the created artifacts are stored in `navigator_workspace` and a Package object is
returned from the function. Read more about optimize
in the [documentation](models_optimize/optimize/optimize.md).

## Deploy the model in PyTriton

The [PyTriton](https://github.com/triton-inference-server/pytriton) can be used to serve inference of any optimized
format. The Triton Model Navigator provides a dedicated `PyTritonAdapter` to retrieve the `runner` and other information required
to bind a model for serving inference. The `runner` is an abstraction that connects the model checkpoint with its
runtime, making the inference process more accessible and straightforward.

Following that, you can initialize the PyTriton server using the adapter information:

```python
pytriton_adapter = nav.pytriton.PyTritonAdapter(package=package, strategy=nav.MaxThroughputStrategy())
runner = pytriton_adapter.runner

runner.activate()


@batch
def infer_func(**inputs):
    return runner.infer(inputs)


with Triton() as triton:
    triton.bind(
        model_name="resnet50",
        infer_func=infer_func,
        inputs=pytriton_adapter.inputs,
        outputs=pytriton_adapter.outputs,
        config=pytriton_adapter.config,
    )
    triton.serve()
```

Read more about deploying the model on PyTriton
in the [documentation](inference_deployment/pytriton/deployment.md).

## Deploy model in Triton Inference Server

The optimized model can also be used for serving inference
on the [Triton Inference Server](https://github.com/triton-inference-server/server) when the serialized format has been
created. The Triton Model Navigator provides functionality to generate a model deployment configuration directly inside
Triton `model_repository`. The following command will select the
model format with the highest throughput and create the Triton deployment in the defined path to the model repository:

```python
nav.triton.model_repository.add_model_from_package(
    model_repository_path=pathlib.Path("model_repository"),
    model_name="resnet50",
    package=package,
    strategy=nav.MaxThroughputStrategy(),
)
```

Once the entry is created, you can simply
start [Triton Inference Server](https://github.com/triton-inference-server/server),
mounting the defined `model_repository_path`.

Read more about deploying the model on the Triton Inference Server
in the [documentation](https://triton-inference-server.github.io/model_navigator/latest/inference_deployment/triton/deployment/).

## Using the Navigator Package

The `Navigator Package` is an artifact that can be produced at the end of the optimization process. The package is a
simple
zip file that contains the optimization details, model metadata. Serialized formats and can be saved using:

```python
nav.package.save(
    package=package,
    path="/path/to/package.nav"
)
```

The package can be easily loaded on other machines and used to re-run the optimization process or profile the model.
Read
more about using the package in the [documentation](models_optimize/package/package.md).

## Profile any model or callable in Python

The Triton Model Navigator enhances models and pipelines and provides a uniform method for profiling any Python
function, callable, or model. At present, our support is limited strictly to static batch profiling scenarios.

As an example, we will use a simple function that simply sleeps for 50 ms:

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

Finally, run the profiling of the function with the prepared dataloader:

```python
nav.profile(custom_fn, dataloader)
```
