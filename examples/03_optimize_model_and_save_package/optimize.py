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

import argparse
from typing import Iterable

import numpy as np
import torch  # pytype: disable=import-error

import model_navigator as nav
from model_navigator.api.config import Sample


def get_model():
    """Returns a simple torch.nn.Linear model"""
    return torch.nn.Linear(5, 7).eval()


def get_dataloader():
    """Returns a ramdom dataloader containing 10 batches of 3x5 tensors"""
    return [torch.randn(3, 5) for _ in range(10)]


def get_verify_function():
    """Define verify function that compares outputs of the torch model and the optimized model."""

    def verify_func(ys_runner: Iterable[Sample], ys_expected: Iterable[Sample]) -> bool:
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.allclose(a, b, rtol=1.0e-3, atol=1.0e-3) for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="linear.nav",
    )
    return parser.parse_args()


def main():
    """Get model, dataloader, verify_func, run optimization and save package"""
    args = parse_args()

    model = get_model()
    dataloader = get_dataloader()
    verify_func = get_verify_function()

    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_func,  # verify_func is optional but recommended.
    )

    """
    Saves the package to the specified path.

    This proces create zipped directory with `.nav` extension.
    It contains all the information about performed optimizations and the models itself.

    Only base models and those with minimal latency and maximal throughput are saved.

    Base formats are:
    Torch:
        - TorchScript
        - ONNX
    TensorFlow2:
        - SavedModel
    JAX:
        - SavedModel
    ONNX:
        - ONNX
    """
    nav.package.save(package, args.output_path, override=True)  # override the existing package if it exists.


if __name__ == "__main__":
    main()
