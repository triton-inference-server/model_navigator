#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""e2e tests for exporting PyTorch identity model"""

import argparse
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_STATUES = [
    "onnx.OnnxCUDA",
    "onnx.OnnxTensorRT",
    "torch.TorchCUDA",
    "torchscript-script.TorchScriptCUDA",
    "torchscript-trace.TorchScriptCUDA",
    "trt-fp16.TensorRT",
    "trt-fp32.TensorRT",
]


def main():
    import numpy as np
    import torch  # pytype: disable=import-error

    import model_navigator as nav
    from tests import utils
    from tests.functional.common.utils import (
        collect_expected_files,
        collect_optimize_status,
        validate_package,
        validate_status,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Timeout for test.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=utils.DEFAULT_LOG_FORMAT)
    LOGGER.debug(f"CLI args: {args}")

    class IdentityWithUnusedInput(torch.nn.Module):
        def forward(self, x, y):
            return x

    model = IdentityWithUnusedInput()
    dataloader = [{"x": torch.randn(2, 3), "y": torch.randn(2, 1)} for _ in range(2)]
    performance_dataloader = [{"x": torch.randn(2, 3), "y": torch.randn(2, 1)} for _ in range(2)]

    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(np.allclose(a, b) for a, b in zip(y_runner.values(), y_expected.values())):
                return False
        return True

    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_func,
        input_names=("input_0", "input_1"),
        verbose=True,
        optimization_profile=nav.OptimizationProfile(
            batch_sizes=[1, 8, 16], stability_percentage=100, dataloader=performance_dataloader
        ),
        custom_configs=[nav.OnnxConfig(opset=14)],
    )
    package_path = pathlib.Path("package.nav")
    nav.package.save(package, package_path)

    status_file = args.status
    status = collect_optimize_status(package.status)

    validate_status(status, expected_statuses=EXPECTED_STATUES)

    expected_files = collect_expected_files(package_path=package_path, status=package.status)
    validate_package(package_path=package_path, expected_files=expected_files)

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
