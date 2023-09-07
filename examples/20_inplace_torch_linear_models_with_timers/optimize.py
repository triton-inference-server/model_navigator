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
import pathlib

import torch  # pytype: disable=import-error

import model_navigator as nav

nav.inplace_config.mode = os.environ.get("MODEL_NAVIGATOR_INPLACE_MODE", nav.inplace_config.mode)
nav.inplace_config.min_num_samples = int(
    os.environ.get("MODEL_NAVIGATOR_MIN_NUM_SAMPLES", nav.inplace_config.min_num_samples)
)

LOGGER = logging.getLogger("model_navigator.inplace")


def get_model():
    """Returns a simple torch.nn.Linear model"""
    return torch.nn.Linear(5, 5).eval()


def get_dataloader():
    """Returns a ramdom dataloader containing 10 batches of 3x5 tensors"""
    return [torch.randn(1, 5) for _ in range(10)]


def get_pipeline():
    class LinearPipeline:
        def __init__(self):
            self.model_a = get_model()
            self.model_b = get_model()
            self.model_c = get_model()

        def __call__(self, x):
            output_a = self.model_a(x)
            output_b = self.model_b(output_a)
            output_c = self.model_c(output_b)
            return output_c

    return LinearPipeline()


def main():
    """Get pipeline, dataloader and run inplace optimization."""

    info = {
        "name": "torch-linear",
        "source": "examples",
        "repository": "https://github.com/triton-inference-server/model_navigator/tree/main/examples",
        "PIC": "name",
        "comment": "single layer demo model",
    }

    timer = nav.Timer(name="torch-linear", info=info)
    pipeline = get_pipeline()

    pipeline.model_a = nav.Module(pipeline.model_a, name="linear_a", timer=timer)

    pipeline.model_b = nav.Module(pipeline.model_b, name="linear_b", timer=timer)

    pipeline.model_c = nav.Module(
        pipeline.model_c,
        name="linear_c",
        timer=timer,
    )

    dataloader = get_dataloader()

    for _ in range(10):
        with timer:
            _ = pipeline(dataloader[0])

    original = pathlib.Path("./original-time-data.yaml")
    optimized = pathlib.Path("./optimized-time-data.yaml")

    if nav.inplace_config.mode == nav.Mode.PASSTHROUGH:
        timer.save("./original-time-data.yaml")
    elif nav.inplace_config.mode == nav.Mode.RUN:
        timer.save("./optimized-time-data.yaml")
        if original.exists() and optimized.exists():
            cmp = nav.TimerComparator(original, optimized)
            LOGGER.info(cmp.get_report())


if __name__ == "__main__":
    main()
