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

The [Triton Model Navigator](https://github.com/triton-inference-server/model_navigator) automates
the process of moving model from source to deployment on NVIDIA Triton Inference Server. The tool validate possible
export and conversion paths to serializable formats like NVIDIA TensorRT and select the most promising format for
production deployment.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Installation](#installation)
- [How it works?](#how-it-works)
- [Quick Start](#quick-start)
  - [Export and optimize model](#export-and-optimize-model)
  - [NVIDIA Triton Inference Server deployment](#nvidia-triton-inference-server-deployment)
- [Examples](#examples)
  - [Optimize for various frameworks](#optimize-for-various-frameworks)
  - [Optimize Navigator Package](#optimize-navigator-package)
  - [Using model on Triton Inference Server](#using-model-on-triton-inference-server)
- [Documentation](#documentation)
- [Useful Links](#useful-links)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Installation

The following prerequisites must be fulfilled to use Triton Model Navigator

- Installed Python `3.8+`
- Installed [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for TensorRT models export.

We recommend to use NGC Containers for PyTorch and TensorFlow which provide have all necessary dependencies:

- [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

To install Triton Model Navigator from source use pip command:

```shell
$ pip install --extra-index-url https://pypi.ngc.nvidia.com .[<extras,>]
```

Extras:

- tensorflow - Model Navigator for TensorFlow2
- jax - Model Navigator for JAX

## How it works?

The Triton Model Navigator is designed to provide a single entrypoint for each supported framework. The usage is
simple as call a dedicated `optimize` function to start the process of searching for the best
possible deployment by going through a broad spectrum of model conversions.

The `optimize` internally it performs model export, conversion, correctness testing, performance profiling,
and saves all generated artifacts in the `navigator_workspace`, which is represented by a returned `package` object.
The result of `optimize` process can be saved as a portable Navigator Package with the `save` function.
Saved packages only contain the base model formats along with the best selected format based on latency and throughput.
The package can be reused to recreate the process on same or different hardware. The configuration and execution status
is saved in the `status.yaml` file located inside the workspace and the `Navigator Package`.

Finally, the `Navigator Packge` can be used for model deployment
on [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server). Dedicated API helps with obtaining all
necessary parameters and creating `model_repository` or receive the optimized model for inference in Python environment.

## Quick Start

The quick start presents how to optimize Python model for deployment on Triton Inference Server. In the
example we are using a simple TensorFlow 2 model.

### Export and optimize model

To use Triton Model Navigator you must prepare model and dataloader. We recommend to create following helper
functions:

- `get_model` - return model object
- `get_dataloader` - generate samples required for export and conversion
- `get_verify_func` (optional) - validate the correctness of models based on implemented metric

Next you can use Triton Model Navigator `optimize` function with provided model, dataloader and verify function
to export and convert model to all supported formats.

See the below example of optimizing a simple TensorFlow model.

```python
import logging

import numpy as np
import tensorflow as tf

import model_navigator as nav

# enable tensorflow memory growth to avoid allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

LOGGER = logging.getLogger(__name__)


# dataloader is used for inference and finding input shapes of the model.
# If you do not have dataloader, create one with samples with min and max shapes.
def get_dataloader():
    return [np.random.rand(1, 224, 224, 3).astype("float32") for _ in range(10)]


def get_verify_function():
    def verify_func(ys_runner, ys_expected):
        for a, b in zip(ys_runner, ys_expected):
            if not (np.isclose(a["output__0"], b["output__0"], atol=0.01)).all():
                return False

        return True

    return verify_func


# Model inputs must be a Tensor to support deployment on Triton Inference Server.
def get_model():
    inp = tf.keras.layers.Input((224, 224, 3))
    layer_output = tf.keras.layers.Lambda(lambda x: x)(inp)
    layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
    layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
    layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
    layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
    model_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
    return tf.keras.Model(inp, model_output)

# Check documentation for more details about Profiler Configuration options.
def get_profiler_config():
    return nav.ProfilerConfig()


model = get_model()
dataloader = get_dataloader()
verify_func = get_verify_function()
profiler_config = get_profiler_config()

# Model Navigator optimize starts export, optimization and testing process.
# The resulting package represents all artifacts produced by Model Navigator.
package = nav.tensorflow.optimize(
    model=model,
    profiler_config=profiler_config,
    target_formats=(nav.Format.ONNX,),
    dataloader=dataloader,
    verify_func=verify_func,
)

# Save nav package that can be used for Triton Inference Server deployment or obtaining model runner later.
# The package contains base format checkpoints that can be used for all other conversions.
# Models with minimal latency and maximal throughput are added to the package.
nav.package.save(package=package, path="mlp.nav")
```

You can customize behavior of export and conversion steps
passing [CustomConfig][model_navigator.api.config.CustomConfig]
to `optimize` function.

### NVIDIA Triton Inference Server deployment

If you prefer the standalone [NVIDIA Triton Inference Server](https://github.com/triton-inference-server) you can create
and use `model_repository`.

```python
import logging
import pathlib

from model_navigator.exceptions import ModelNavigatorEmptyPackageError, ModelNavigatorError,
    ModelNavigatorWrongParameterError
import model_navigator as nav

LOGGER = logging.getLogger(__name__)

package = nav.package.load("mlp.nav", "load_workspace")

# Create model_repository for standalone Triton deployment
try:
    nav.triton.model_repository.add_model_from_package(
        model_repository_path=pathlib.Path("model_repository"), model_name="dummy_model", package=package
    )
except (ModelNavigatorWrongParameterError, ModelNavigatorEmptyPackageError, ModelNavigatorError) as e:
    LOGGER.warning(f"Model repository cannot be created.\n{str(e)}")
```

Use command to start server with provided `model_repository`:

```shell
$ docker run --gpus=1 --rm \
  -p8000:8000 \
  -p8001:8001 \
  -p8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.01-py3 \
  tritonserver --model-repository=/models
```

## Examples

We provide simple examples how to use Triton Model Navigator to optimize the PyTorch, TensorFlow2, JAX and ONNX models
for deployment on Triton Inference Server.

### Optimize for various frameworks

- `PyTorch`:
  * [Linear Model](examples/torch/linear)
  * [ResNet50](examples/torch/resnet50)
  * [BERT](examples/torch/bert)

- `TensorFlow`:
  * [Linear Model](examples/tensorflow/linear)
  * [EfficientNet](examples/tensorflow/efficientnet)
  * [BERT](examples/tensorflow/bert)

- `JAX`:
  * [Linear Model](examples/jax/linear)
  * [GPT-2](examples/jax/gpt2)

- `ONNX`:
  * [Identity Model](examples/onnx/identity)

### Optimize Navigator Package

The Navigator Package can be reused for optimize e.g. on the new hardware or with newer libraries.
The example code can be found in [examples/package](examples/package).

### Using model on Triton Inference Server

The optimized model by Triton Model Navigator can be used for serving inference through Triton Inference Server. The
example code can be found in [examples/triton](examples/triton).


## Documentation

More information about `optimize` function, working with packages and Triton Inference Server can be
found in [documentation](https://triton-inference-server.github.io/triton_model_navigator).

## Useful Links

* [Changelog](CHANGELOG.md)
* [Support Matrix](docs/support_matrix.md)
* [Known Issues](docs/known_issues.md)
* [Contributing](CONTRIBUTING.md)
