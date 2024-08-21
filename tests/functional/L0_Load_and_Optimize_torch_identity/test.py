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
import pathlib

import yaml
from loguru import logger

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_STATUSES = [
    "onnx.OnnxCUDA",
    "onnx.OnnxTensorRT",
    "torchscript-script.TorchScriptCUDA",
    "torchscript-trace.TorchScriptCUDA",
    "torch-exportedprogram.TorchExportedProgramCUDA",
    "torch-trt-fp16.TorchTensorRT",
    "torch-trt-fp32.TorchTensorRT",
    "trt-fp16.TensorRT",
    "trt-fp32.TensorRT",
]


def main():
    import model_navigator as nav
    from model_navigator.configuration import DEFAULT_TORCH_TARGET_FORMATS
    from tests.functional.common.utils import collect_optimize_status, validate_status
    from tests.utils import get_assets_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )

    args = parser.parse_args()

    logger.debug(f"CLI args: {args}")

    package_path = get_assets_path() / "packages" / "torch_identity.nav"
    package = nav.package.load(package_path)

    new_package = nav.package.optimize(
        package,
        verbose=True,
        target_formats=DEFAULT_TORCH_TARGET_FORMATS,
        optimization_profile=nav.OptimizationProfile(batch_sizes=[1, 32], stability_percentage=100),
    )

    status_file = args.status
    status = collect_optimize_status(new_package.status)

    validate_status(status, expected_statuses=EXPECTED_STATUSES)

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
