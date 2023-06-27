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

# Quick Start

These sections provide an overview of optimizing the model, deploying model for serving inference
on [PyTriton](https://github.com/triton-inference-server/pytriton)
or [Triton Inference Server](https://github.com/triton-inference-server/server)
as well as using the Navigator Package. In each section you will find links to learn more about Model Navigator
features.

## Optimize Model

Optimizing models using Model Navigator is as simply as calling `optimize` function. The optimization process requires
at least:

- `model` - a Python object, callable or file path with model to optimize.
- `dataloader` - a method or class generating input data. The data is utilized to determine the maximum and minimum
  shapes
  of the model inputs and create output samples that are used during the optimization process.

Here is an example of running `optimize` on Torch Hub ResNet50 model:

```python
import torch
import model_navigator as nav

package = nav.torch.optimize(
    model=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).eval(),
    dataloader=[torch.randn(1, 3, 256, 256) for _ in range(10)],
)
```

Once the model has been optimized the created artifacts are stored in `navigator_workspace` and a Package object is
returned from the function. Read more about optimize
in [documentation](optimize/optimize.md)

## Deploy model in PyTriton

The [PyTriton](https://github.com/triton-inference-server/pytriton) can be used to serve inference of any optimized
format. Model Navigator provide a dedicated `PyTritonAdapter` to retrieve the `runner` and other information required
to bind model for serving inference. The `runner` is an abstraction that connects the model checkpoint with its
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

Read more about deploying model on PyTriton
in [documentation](pytriton/pytriton_deployment.md)

## Deploy model in Triton Inference Server

The optimized model can be also used for serving inference
on [Triton Inference Server](https://github.com/triton-inference-server/server) when the serialized format has been
created. Model Navigator provide functionality to generate a model deployment configuration directly inside
Triton `model_repository`. The following command will select the
model format with the highest throughput and create the Triton deployment in defined path to model repository:

```python
nav.triton.model_repository.add_model_from_package(
    model_repository_path=pathlib.Path("model_repository"),
    model_name="resnet50",
    package=package,
    strategy=nav.MaxThroughputStrategy(),
)
```

Once the entry is created, you can simply
start [Triton Inference Server](https://github.com/triton-inference-server/server)
mounting the defined `model_repository_path`.

Read more about deploying model on Triton Inference Server
in [documentation](https://triton-inference-server.github.io/model_navigator/latest/triton/triton_deployment/)

## Using Navigator Package

The `Navigator Package` is an artifact that can be produced at the end of optimization process. The package is a simple
Zip file which contains the optimization details, model metadata and serialized formats and can be saved using:

```python
nav.package.save(
    package=package,
    path="/path/to/package.nav"
)
```

The package can be easily loaded on other machine and used to re-run the optimization process or profile the model. Read
more about using pacakge
in [documentation](package/package.md).
