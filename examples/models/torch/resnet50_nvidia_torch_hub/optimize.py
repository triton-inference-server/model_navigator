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
    resnet50 = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_resnet50", pretrained=True)
    resnet50.eval()
    return resnet50


def get_dataloader():
    """Returns a ramdom dataloader"""
    return [torch.randn(1, 3, 256, 256) for _ in range(10)]


def main():
    """Get model, dataloader, and run optimization"""
    model = get_model()
    dataloader = get_dataloader()

    """
    Optimize the model by performing model export, conversion, correctness tests, and profiling.

    Results are saved in `navigator_workspace` directory.
    """
    nav.torch.optimize(
        model=model,
        dataloader=dataloader,
    )


if __name__ == "__main__":
    main()
