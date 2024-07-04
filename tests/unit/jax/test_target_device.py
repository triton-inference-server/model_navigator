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
from importlib.util import find_spec

import jax.numpy as jnp  # pytype: disable=import-error
import numpy
import pytest
import tensorflow  # pytype: disable=import-error

import model_navigator as nav
from model_navigator.exceptions import ModelNavigatorConfigurationError
from tests.utils import gpu_count

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

dataloader = [numpy.random.rand(1, 10, 10) for _ in range(5)]
params = numpy.random.rand(1, 10, 10)


def predict(inputs, params):
    outputs = jnp.dot(inputs, params)
    return outputs


@pytest.mark.skipif(gpu_count() == 0 or not find_spec("jax"), reason="GPU is not available or JAX is not installed.")
def test_raise_error_when_target_device_cpu_and_gpu_available():
    # Check if GPU is available
    with pytest.raises(ModelNavigatorConfigurationError):
        nav.experimental.jax.optimize(
            model=predict,
            model_params=params,
            dataloader=dataloader,
            target_device=nav.DeviceKind.CPU,
        )


def test_raise_error_when_target_device_cuda_and_savedmodel_cpu_runner_passed():
    with pytest.raises(ModelNavigatorConfigurationError):
        nav.experimental.jax.optimize(
            model=predict,
            model_params=params,
            dataloader=dataloader,
            target_device=nav.DeviceKind.CUDA,
            runners=("TensorFlowSavedModelCPU",),
        )
