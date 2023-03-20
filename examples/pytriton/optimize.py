# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export script template for Model Navigator"""
import logging

import numpy as np
import tensorflow as tf  # pytype: disable=import-error

import model_navigator as nav

# enable tensorflow memory growth to avoid allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

LOGGER = logging.getLogger(__name__)


def get_dataloader():
    """Function returning dataloader that will be used by Model Navigator Optimize function.

    Returns:
        Sizeable, finite iterable.
    """
    return [np.random.rand(1, 224, 224, 3).astype("float32") for _ in range(10)]


def get_verify_function():
    """Returns verify_func that will be used for additional model verification.

    verify_func must take two arguments of type Sequence.
    First argument represents outputs inferred by converted model.
    Second argument represents outputs inferred from source framework model.

    Passed outputs can be used to calculate custom metric.

    Returns:
        True if verifications is successful, otherwise False.

    """

    def verify_func(ys_runner, ys_expected):
        for a, b in zip(ys_runner, ys_expected):
            if not (np.isclose(a["output__0"], b["output__0"], atol=0.01)).all():
                return False

        return True

    return verify_func


def get_model():
    """Function returns model object, model path or infer function depending on framework.

    Returns:
        For TensorFlow2: tensorflow.keras.Model
        For PyTorch: torch.nn.Module
        For ONNX: Union[Path, str]
        For JAX: Callable
    """
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


# nav.<framework>.optimize starts Model Navigator and perform all steps automatically:
# export, conversion, correctness tests, performance profiling, model verification with verify_func.

# Returns:
#   package describing optimize status and all results.
#   It can be latter used to get inference runners or Triton Model Store.
package = nav.tensorflow.optimize(
    model=model,
    runners=(
        "OnnxCUDA",
        "OnnxTensorRT",
        "TensorRT",
        "TensorFlowSavedModelCUDA",
        "TensorFlowTensorRT",
    ),
    profiler_config=profiler_config,
    dataloader=dataloader,
    verify_func=verify_func,
    verbose=True,
)

nav.package.save(package=package, path="mlp.nav", override=True)
