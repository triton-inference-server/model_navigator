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


def get_model():
    """Returns a simple torch.nn.Linear model"""
    return torch.nn.Linear(5, 7).eval()


def get_dataloader():
    """Returns a ramdom dataloader containing 10 batches of 3x5 tensors"""
    return [torch.randn(3, 5) for _ in range(10)]


def get_configuration():
    """Returns a configuration with custom arguments for Torch-TensorRT and TensorRT conversions"""
    return {
        "input_names": ("input",),
        "custom_configs": [
            nav.TorchTensorRTConfig(custom_args={"ir": "ts"}),
            nav.TensorRTConfig(custom_args={"--shape-inference": True}),
        ],
    }


def main():
    """Get model, dataloader, custom configuration, and run optimization"""
    model = get_model()
    dataloader = get_dataloader()
    configuration = get_configuration()

    """
    Optimize the model by performing model export, conversion, correctness tests, and profiling.

    Results are saved in `navigator_workspace` directory.
    """
    nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        **configuration,
    )


if __name__ == "__main__":
    main()
