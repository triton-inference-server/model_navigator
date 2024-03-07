#!/usr/bin/env python3
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
import torch  # pytype: disable=import-error

import model_navigator as nav

device = torch.device("cuda:0")
model_a = nav.Module(torch.nn.Linear(2, 2, device=device), name="model_a")
model_b = nav.Module(torch.nn.Linear(2, 2, device=device), name="model_b")
model_c = nav.Module(torch.nn.Linear(2, 2, device=device), name="model_c")


def call(input_):
    x = model_a(input_)
    x = model_b(x)
    x = model_c(x)
    return x


config = nav.OptimizeConfig(
    target_formats=(nav.Format.ONNX, nav.Format.TORCH, nav.Format.TENSORRT),
    runners=("OnnxCUDA", "TorchCUDA", "TensorRT"),
    optimization_profile=nav.OptimizationProfile(max_batch_size=8),
)

dataloader = [(1, torch.randn(1, 2, device=device))]

nav.optimize(
    call,
    dataloader,
    config,
)

nav.profile(
    call,
    dataloader,
    target_formats=(nav.Format.ONNX, nav.Format.TORCH),
    runners=("OnnxCUDA", "TorchCUDA"),
)
