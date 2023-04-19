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

import pytest
import tensorflow  # pytype: disable=import-error

import model_navigator as nav
from model_navigator.exceptions import ModelNavigatorConfigurationError

VALUE_IN_TENSOR = 9.0
dataloader = [tensorflow.fill(dims=[1, 224, 224, 3], value=VALUE_IN_TENSOR) for _ in range(5)]


inp = tensorflow.keras.layers.Input((224, 224, 3), name="input__1")
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)

model = tensorflow.keras.Model(inp, model_output)


def test_raise_error_when_target_device_cpu_and_gpu_available():
    # Check if GPU is available
    if any(device.device_type == "GPU" for device in tensorflow.config.get_visible_devices()):
        with pytest.raises(ModelNavigatorConfigurationError):
            nav.tensorflow.optimize(
                model=model,
                dataloader=dataloader,
                target_device=nav.DeviceKind.CPU,
            )


def test_raise_error_when_target_device_cuda_and_savedmodel_cpu_runner_passed():
    with pytest.raises(ModelNavigatorConfigurationError):
        nav.tensorflow.optimize(
            model=model, dataloader=dataloader, target_device=nav.DeviceKind.CUDA, runners=("TensorFlowSavedModelCPU",)
        )
