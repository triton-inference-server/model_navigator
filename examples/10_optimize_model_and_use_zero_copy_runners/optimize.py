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
import time

import torch  # pytype: disable=import-error

import model_navigator as nav

logging.basicConfig(format="%(message)s", level=logging.INFO)


def get_model():
    """Returns a simple torch.nn.Linear model"""
    return torch.nn.Linear(1024, 1024).eval()


def get_dataloader():
    """Returns a random dataloader containing a single sample"""
    return [torch.randn(64 * 1024, 1024)]


def main():
    """Get model, dataloader, and run optimization"""
    model = get_model()
    dataloader = get_dataloader()
    """
    Optimize the model by performing model export, conversion, correctness tests,
    profiling and additional verification.
    """
    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        batching=False,  # to avoid batch analysis for the simplicity of the example
    )

    """
    By default runners accept numpy arrays as input and return numpy arrays as output.
    Sometimes it is more convenient to use CUDA tensors as input and output to avoid copying data to and from CPU.
    This can be achieved by passing a `return_type` parameter to the `get_runner` method.
    Here we compare the runtime of the default runner and the runner that returns CUDA tensors when running inference in a loop.
    """
    numpy_runner = package.get_runner()
    numpy_feed_dict = {"input__0": dataloader[0].cpu().detach().numpy()}
    start_time = time.monotonic()
    with numpy_runner:
        for _ in range(100):
            out = numpy_runner.infer(numpy_feed_dict, check_inputs=True)
            numpy_feed_dict = {"input__0": out["output__0"]}
    numpy_runtime = time.monotonic() - start_time

    torch_runner = package.get_runner(return_type=nav.TensorType.TORCH)
    torch_feed_dict = {"input__0": dataloader[0].cuda()}
    start_time = time.monotonic()
    with torch_runner:
        for _ in range(100):
            out = torch_runner.infer(torch_feed_dict, check_inputs=True)
            torch_feed_dict = {"input__0": out["output__0"]}
    torch_runtime = time.monotonic() - start_time

    logging.info(f"Runtime of the default runner: {numpy_runtime:.3f} seconds")
    logging.info(f"Runtime of the zero-copy runner: {torch_runtime:.3f} seconds")


if __name__ == "__main__":
    main()
