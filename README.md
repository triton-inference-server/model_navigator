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

# Triton Model Navigator

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Documentation](#documentation)
- [Support Matrix](#support-matrix)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Optimize Model](#optimize-model)
  - [Deploy model in PyTriton](#deploy-model-in-pytriton)
  - [Deploy model in Triton Inference Server](#deploy-model-in-triton-inference-server)
  - [Using Navigator Package](#using-navigator-package)
  - [Inplace Optimize (alpha)](#inplace-optimize-alpha)
    - [Stable Diffusion example](#stable-diffusion-example)
  - [Profiling](#profiling)
- [Examples](#examples)
- [Useful Links](#useful-links)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

Model optimization plays a crucial role in unlocking the maximum performance capabilities of the underlying hardware. By
applying various transformation techniques, models can be optimized to fully utilize the specific features offered by
the hardware architecture to improve the inference performance and cost. Furthermore, in many cases allow for
serialization of models, separating them from the source code. The serialization process enhances portability, allowing
the models to be seamlessly deployed in production environments. The decoupling of models from the source code also
facilitates maintenance, updates, and collaboration among developers. However, this process comprises multiple steps and
offers various potential paths, making manual execution complicated and time-consuming.

The [Triton Model Navigator](https://github.com/triton-inference-server/model_navigator) offers a user-friendly and
automated solution for optimizing and deploying machine learning models. Using a single entry point for
various supported frameworks, allowing users to start the process of searching for the best deployment option with a
single call to the dedicated `optimize` function. Model Navigator handles model export, conversion, correctness testing,
and profiling to select optimal model format and save generated artifacts for inference deployment on the
[PyTriton](https://github.com/triton-inference-server/pytriton)
or [Triton Inference Server](https://github.com/triton-inference-server/server).

The high-level flowchart below illustrates the process of moving models from source code to deployment optimized formats
with the support of the Model Navigator:

![Overview](docs/assets/overview.svg)

## Documentation

The full documentation about optimizing models, using Navigator Package and deploying models in PyTriton and/or Triton
Inference Server can be found in [documentation](https://triton-inference-server.github.io/model_navigator).

## Support Matrix

The Model Navigator generates multiple optimized and production-ready models. The table below illustrates the model
formats that can be obtained by using the Model Navigator with various frameworks.

**Table:** Supported conversion target formats per each supported Python framework or file.

| **PyTorch**        | **TensorFlow 2**       | **JAX**                | **ONNX** |
|--------------------|------------------------|------------------------|----------|
| Torch Compile      | SavedModel             | SavedModel             | TensorRT |
| TorchScript Trace  | TensorRT in TensorFlow | TensorRT in TensorFlow |          |
| TorchScript Script | ONNX                   | ONNX                   |          |
| Torch-TensorRT     | TensorRT               | TensorRT               |          |
| ONNX               |                        |                        |          |
| TensorRT           |                        |                        |          |

**Note:** The Model Navigator has the capability to support any Python function as input. However, in this particular
case, its role is limited to profiling the function without generating any serialized models.

The Model Navigator stores all artifacts within the `navigator_workspace`. Additionally, it provides an option to save
a portable and transferable `Navigator Package` - an artifact that includes only the models with minimal latency and
maximal throughput. This package also includes base formats that can be used to regenerate the `TensorRT` plan on the
target hardware.

**Table:** Model formats that can be generated from saved `Navigator Package` and from model sources.

| **From model source** | **From Navigator Package** |
|-----------------------|----------------------------|
| SavedModel            | TorchTensorRT              |
| TensorFlowTensorRT    | TensorRT in TensorFlow     |
| TorchScript Trace     | ONNX                       |
| TorchScript Script    | TensorRT                   |
| Torch 2 Compile       |                            |
| TorchTensorRT         |                            |
| ONNX                  |                            |
| TensorRT              |                            |

## Installation

The following prerequisites must be fulfilled to use Triton Model Navigator

- Installed Python `3.8+`
- Installed [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for TensorRT models export.

We recommend to use NGC Containers for PyTorch and TensorFlow which provide have all necessary dependencies:

- [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

The package can be installed from `pypi.org` using extra index url:

```shell
pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[<extras,>]
```

or with nvidia-pyindex:

```shell
pip install nvidia-pyindex
pip install -U triton-model-navigator[<extras,>]
```

To install Triton Model Navigator from source use pip command:

```shell
$ pip install --extra-index-url https://pypi.ngc.nvidia.com .[<extras,>]
```

Extras:

- `tensorflow` - Model Navigator with dependencies for TensorFlow2
- `jax` - Model Navigator with dependencies for JAX

For using with PyTorch no extras are needed.

## Quick Start

This sections describe simple steps of optimizing the model for serving inference on PyTriton or Triton Inference Server
as well as saving a Navigator Package for distribution.

### Optimize Model

Optimizing models using Model Navigator is as simply as calling `optimize` function. The optimization process requires
at least:

- `model` - a Python object, callable or file path with model to optimize.
- `dataloader` - a method or class generating input data. The data is utilized to determine the maximum and minimum
  shapes of the model inputs and create output samples that are used during the optimization process.

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
in [documentation](https://triton-inference-server.github.io/model_navigator/latest/package/package/)

### Deploy model in PyTriton

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
in [documentation](https://triton-inference-server.github.io/model_navigator/latest/pytriton/pytriton_deployment/)

### Deploy model in Triton Inference Server

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
mounting the defined `model_repository_path`. Read more about deploying model on Triton Inference Server
in [documentation](https://triton-inference-server.github.io/model_navigator/latest/triton/triton_deployment/)

### Using Navigator Package

The `Navigator Package` is an artifact that can be produced at the end of the optimization process. The package is a simple
Zip file which contains the optimization details, model metadata and serialized formats and can be saved using:

```python
nav.package.save(
    package=package,
    path="/path/to/package.nav"
)
```

The package can be easily loaded on other machines and used to re-run the optimization process or profile the model. Read
more about using package
in [documentation](https://triton-inference-server.github.io/model_navigator/latest/package/package/).

### Inplace Optimize (alpha)

The Inplace Optimize is a powerful tool that allows seamless optimization of models for deployment, such as converting them to TensorRT, without requiring any changes to the original Python pipelines.
All that is required is to wrap a module with a single line of code:

```python
import model_navigator as nav

dataloader = [(1, "<a photo of an astronaut riding a horse on mars>")] # batch_size, sample

pipeline = Pipeline(...)
pipeline.model = nav.Module(pipeline.model)

nav.optimize(pipeline, dataloader)

pipeline(...)

```

Inplace Optimize is currently in alpha and not all modules might be supported. Specifically, modules with data dependent control flow are currently not supported.

#### Stable Diffusion example

The provided [example](examples/13_inplace_stable_diffusion/) demonstrates how to use the API to optimize a Stable Diffusion Pipeline model by converting it to TensorRT, thereby accelerating its inference on NVIDIA GPUs.


### Profiling

Model Navigator provides `nav.profile` functionality that helps with meaasuring performance of any Python function or pipelines optimized with Inplace Optimize functionality

```python
nav.profile(func=pipeline, dataloader)
```

This code will profile provided callable with samples from dataloader. In case of pipeline optimized with Inplace Optimize, it will iterate all successfully exported and converted formats.

Profiling results will be saved in yaml file report.


## Examples

We provide step-by-step [examples](examples) that demonstrate how to use various features of Model Navigator.
For the sake of readability and accessibility, we use a simple `torch.nn.Linear` model as an example.
These [examples](examples) illustrate how to optimize, test and deploy the model on
the [PyTriton](https://github.com/triton-inference-server/pytriton)
and [Triton Inference Server](https://github.com/triton-inference-server/server).

## Useful Links

* [Changelog](CHANGELOG.md)
* [Support Matrix](docs/support_matrix.md)
* [Known Issues](docs/known_issues.md)
* [Contributing](CONTRIBUTING.md)
