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
import torch  # pytype: disable=import-error

import model_navigator as nav
from model_navigator.exceptions import ModelNavigatorConfigurationError

VALUE_IN_TENSOR = 9.0
OPSET = 11

dataloader = [torch.full((1, 1), VALUE_IN_TENSOR) for _ in range(5)]


class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10


model = MyModule()


def test_raise_error_when_target_device_cuda_and_torchscript_cpu_runner_passed():
    with pytest.raises(ModelNavigatorConfigurationError):
        nav.torch.optimize(
            model=model, dataloader=dataloader, target_device=nav.DeviceKind.CUDA, runners=("TorchScriptCPU",)
        )


def test_raise_error_when_target_device_cpu_and_torchscript_cuda_runner_passed():
    with pytest.raises(ModelNavigatorConfigurationError):
        nav.torch.optimize(
            model=model,
            dataloader=dataloader,
            runners=("TorchScriptCUDA",),
            target_device=nav.DeviceKind.CPU,
        )
