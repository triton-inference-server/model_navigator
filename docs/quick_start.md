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

The prerequisite for this section is installing the Triton Model Navigator which can be found
in [installation](installation.md) section.

The quick start presents how to optimize Python model for deployment on Triton Inference Server. In the
example we are using a simple TensorFlow 2 model.

## Export and optimize model

To use Triton Model Navigator you must prepare model and dataloader. We recommend to create following helper
functions:

- `get_model` - return model object
- `get_dataloader` - generate samples required for export and conversion
- `get_verify_func` (optionally) - validate the correctness of models based on implemented metric

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
            if not (a["output__0"] == b["output__0"]).all():
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

## PyTriton deployment

At this point you can use [NVIDIA PyTriton](https://github.com/triton-inference-server/pytriton) for easy deployment
of the exported model. Below you can find an example `serve.py` that will select the best model from a previously
saved `Navigator Package`, get the best runner, and use it to start `PyTriton`.

```python
from pytriton.decorators import batch
from pytriton.triton import Triton

import model_navigator as nav

package = nav.package.load("mlp.nav", "load_workspace")

pytriton_adapter = nav.pytriton.PyTritonAdapter(package=package)
runner = pytriton_adapter.runner
runner.activate()


@batch
def infer_func(**inputs):
    return runner.infer(inputs)


# Connecting inference callback with Triton Inference Server
with Triton() as triton:
    # Load model into Triton Inference Server
    triton.bind(
        model_name="mlp",
        infer_func=infer_func,
        inputs=pytriton_adapter.inputs,
        outputs=pytriton_adapter.outputs,
        config=pytriton_adapter.config,
    )
    # Serve model through Triton Inference Server
    triton.serve()
```

## Triton Inference Server deployment

If you prefer the standalone [NVIDIA Triton Inference Server](https://github.com/triton-inference-server) you can create
and use `model_repository`.

```python
import logging
import pathlib

from model_navigator.exceptions import ModelNavigatorEmptyPackageError, ModelNavigatorError, ModelNavigatorWrongParameterError
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
