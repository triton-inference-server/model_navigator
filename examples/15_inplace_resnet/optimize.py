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

import logging
import os
import time

import torch  # pytype: disable=import-error

import model_navigator as nav

nav.inplace_config.mode = os.environ.get("MODEL_NAVIGATOR_INPLACE_MODE", nav.inplace_config.mode)

LOGGER = logging.getLogger("model_navigator.inplace")
logging.basicConfig(level=logging.INFO)


def get_dataloader():
    return [torch.rand(1, 3, 224, 224, device="cuda") for _ in range(150)]


@nav.module(optimize_config=nav.OptimizeConfig(batching=False))
def get_model():
    return torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).to("cuda").eval()


def main():
    model = get_model()
    dataloader = get_dataloader()

    start = time.monotonic()
    for batch in dataloader:
        model(batch)
    end = time.monotonic()
    LOGGER.info(f"Elapsed time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
