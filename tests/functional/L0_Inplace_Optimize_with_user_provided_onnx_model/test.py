#!/usr/bin/env python3
# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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
import tempfile

import yaml
from loguru import logger

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_PACKAGES = 1
EXPECTED_STATUSES_TEMPLATE = [
    "{name}.{ind}.onnx.OnnxCUDA",
    "{name}.{ind}.onnx.OnnxTensorRT",
    "{name}.{ind}.torch.TorchCUDA",
    "{name}.{ind}.torch.TorchCompileCUDA",
    "{name}.{ind}.trt-fp16.TensorRT",
    "{name}.{ind}.trt-fp32.TensorRT",
]
EXPECTED_STATUSES = [
    status.format(name="torch.nn.modules.linear.Linear", ind=0) for status in EXPECTED_STATUSES_TEMPLATE
]


def main():
    import torch  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.inplace.registry import module_registry
    from tests import utils  # pytype: disable=import-error
    from tests.functional.common.utils import collect_optimize_statuses, validate_status

    def get_model():
        """Returns a simple torch.nn.Linear model"""
        return torch.nn.Linear(5, 7).eval()

    def get_dataloader():
        """Returns a random dataloader containing 10 batches of 3x5 tensors"""
        return [(1, torch.randn(3, 5)) for _ in range(10)]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    args = parser.parse_args()

    model_path = utils.get_assets_path() / "models" / "onnx_from_torch_linear.onnx"
    model = nav.Module(get_model(), model_path=model_path)
    dataloader = get_dataloader()

    optimize_status = nav.optimize(
        func=model,
        dataloader=dataloader,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = pathlib.Path(tmpdir) / "optimized_status.yaml"
        optimize_status.to_file(tmp_file)

    names, packages = [], []
    for name, module in module_registry.items():
        for i, package in enumerate(module.wrapper.packages):
            names.append(f"{name}.{i}")
            packages.append(package)

    assert (
        len(packages) == EXPECTED_PACKAGES
    ), f"Wrong number of packages. Got {len(packages)}. Expected: {EXPECTED_PACKAGES}"

    status = collect_optimize_statuses([package.status for package in packages], names)

    validate_status(status, expected_statuses=EXPECTED_STATUSES)

    status_file = args.status
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
