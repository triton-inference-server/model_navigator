#!/usr/bin/env python3
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

import logging

import torch  # pytype: disable=import-error

import model_navigator as nav

LOGGER = logging.getLogger("model_navigator.inplace")
logging.basicConfig(level=logging.INFO)


def get_dataloader():
    return [(1, torch.rand(1, 3, 224, 224, device="cuda")) for _ in range(150)]


def get_model():
    return torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).to("cuda").eval()


config = nav.OptimizeConfig(
    target_formats=(nav.Format.ONNX, nav.Format.TORCH, nav.Format.TENSORRT),
    runners=("OnnxCUDA", "TorchCUDA", "TensorRT"),
    optimization_profile=nav.OptimizationProfile(max_batch_size=8),
)

model = nav.Module(get_model(), name="resnet18")


def main():
    dataloader = get_dataloader()
    nav.optimize(model, dataloader, config=config)

    nav.profile(model, dataloader)


if __name__ == "__main__":
    main()
