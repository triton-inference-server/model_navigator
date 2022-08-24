# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import jax.numpy as jnp
import numpy
import tensorflow

import model_navigator as nav

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

dataloader = [numpy.random.rand(1, 10, 10) for _ in range(10)]
params = numpy.random.rand(1, 10, 10)


def predict(inputs, params):
    outputs = jnp.dot(inputs, params)
    return outputs


pkg_desc = nav.jax.export(
    model=predict,
    model_params=params,
    dataloader=dataloader,
    override_workdir=True,
    batch_dim=None,
)
