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

import numpy as np

import model_navigator as nav


def get_model():
    def infer_func(input__0):
        return {"output__0": input__0}

    return infer_func


def get_dataloader():
    return [np.random.rand(3, 5).astype("float32") for _ in range(10)]


def get_verify_function():
    """Define verify function that compares outputs of the torch model and the optimized model."""

    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                [np.allclose(a, b, rtol=1.0e-3, atol=1.0e-3) for a, b in zip(y_runner.values(), y_expected.values())]
            ):
                return False
        return True

    return verify_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="identity.nav",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = get_model()
    dataloader = get_dataloader()
    verify_func = get_verify_function()

    package = nav.python.optimize(
        model=model,
        dataloader=dataloader,
        profiler_config=nav.ProfilerConfig(
            batch_sizes=[1, 2, 4],
        ),
        verify_func=verify_func,
    )

    nav.package.save(package, args.output_path, override=True)


if __name__ == "__main__":
    main()
