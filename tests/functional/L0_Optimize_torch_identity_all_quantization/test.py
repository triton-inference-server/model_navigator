#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Test for optimizing PyTorch MLP model with multiple precision types (FP8, INT8, NVFP4)"""

import argparse
import pathlib
import tempfile

import numpy as np
import yaml
from loguru import logger

import model_navigator as nav
from tests.functional.common.utils import collect_optimize_status, validate_status

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_STATUSES = [
    "onnx.OnnxCUDA",
    "onnx.OnnxTensorRT",
    "torch.TorchCUDA",
    "torchscript-script.TorchScriptCUDA",
    "torchscript-trace.TorchScriptCUDA",
    "trt-fp8.TensorRT",
    "trt-int8.TensorRT",
    "trt-nvfp4.TensorRT",
]


def main():
    import torch  # pytype: disable=import-error

    # Define a simple MLP model
    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_size=10, hidden_size=20, output_size=5):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )

    args = parser.parse_args()
    logger.debug(f"CLI args: {args}")

    # Define dimensions for the MLP model
    input_size = 10
    hidden_size = 20
    output_size = 5
    batch_size = 2

    # Create a simple MLP model
    model = SimpleMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # Create test dataloader with appropriate input dimensions
    dataloader = [(batch_size, torch.randn(batch_size, input_size)) for _ in range(2)]

    # Verification function to validate outputs
    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(np.allclose(a, b, rtol=1e-2, atol=1e-2) for a, b in zip(y_runner.values(), y_expected.values())):
                return False
        return True

    optimize_config = nav.OptimizeConfig(
        verbose=True,
        optimization_profile=nav.OptimizationProfile(batch_sizes=[1, 8, 16], stability_percentage=100),
        verify_func=verify_func,
        runners=(
            "OnnxCUDA",
            "OnnxTensorRT",
            "TorchCUDA",
            "TorchScriptCUDA",
            "TensorRT",
        ),
        custom_configs=[
            nav.TensorRTConfig(
                precision=(nav.TensorRTPrecision.INT8, nav.TensorRTPrecision.FP8, nav.TensorRTPrecision.NVFP4),
            ),
        ],
    )
    model = nav.Module(model)
    optimize_status = nav.optimize(func=model, dataloader=dataloader, config=optimize_config)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = pathlib.Path(tmpdir) / "optimized_status.yaml"
        optimize_status.to_file(tmp_file)

    packages = getattr(model._wrapper, "_packages", [])
    assert len(packages) == 1, "Package is not created."
    package = packages[0]

    status_file = args.status
    status = collect_optimize_status(package.status)

    validate_status(status, expected_statuses=EXPECTED_STATUSES)

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
