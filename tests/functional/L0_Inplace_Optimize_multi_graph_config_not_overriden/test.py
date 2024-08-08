#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

EXPECTED_PACKAGES = 2

EXPECTED_STATUSES_TEMPLATE = [
    "{name}.{ind}.trt-fp16.TensorRT",
    "{name}.{ind}.trt-fp32.TensorRT",
]

EXPECTED_STATUSES = [
    status.format(name="identity1", ind=ind) for status in EXPECTED_STATUSES_TEMPLATE for ind in range(1)
] + [status.format(name="identity2", ind=ind) for status in EXPECTED_STATUSES_TEMPLATE for ind in range(1)]


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )

    return parser.parse_args()


def main():
    import torch  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.inplace.registry import module_registry
    from tests.functional.common.utils import collect_optimize_statuses, validate_status

    args = _get_args()

    logger.debug(f"CLI args: {args}")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.mod1 = torch.nn.Identity()
            self.mod2 = torch.nn.Identity()

        def forward(self, x):
            x = self.mod1(x)
            x = self.mod2(x)
            return x

    model = Model()

    dataloader = [
        (1, {"x": torch.randn(1, 3)}),
        (1, {"x": torch.randn(8, 3)}),
        (1, {"x": torch.randn(16, 3)}),
        (1, {"x": torch.randn(16, 3)}),
    ]

    optimize_config = nav.OptimizeConfig(
        verbose=True,
        optimization_profile=nav.OptimizationProfile(batch_sizes=[1, 8, 16], stability_percentage=100),
        verify_func=_verify_func,
        runners=("TensorRT",),
    )
    model.mod1 = nav.Module(model.mod1, name="identity1")
    model.mod2 = nav.Module(model.mod2, name="identity2")

    model.mod1.optimize_config = optimize_config
    model.mod2.optimize_config = optimize_config
    model.mod2.optimize_config.runners = ("TensorRT", "TorchCUDA")

    optimize_status = nav.optimize(func=model, dataloader=dataloader)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = pathlib.Path(tmpdir) / "optimized_status.yaml"
        optimize_status.to_file(tmp_file)

    assert model.mod1.optimize_config.runners == ("TensorRT",), "Optimization should not override the module config."
    assert model.mod2.optimize_config.runners == (
        "TensorRT",
        "TorchCUDA",
    ), "Optimization should not override the module config."

    names, packages = [], []
    for name, module in module_registry.items():
        for i, package in enumerate(module.wrapper.packages):
            names.append(f"{name}.{i}")
            packages.append(package)
    assert (
        len(packages) == EXPECTED_PACKAGES
    ), f"Wrong number of packages. Got {len(packages)}. Expected: {EXPECTED_PACKAGES}"

    status_file = args.status
    status = collect_optimize_statuses([package.status for package in packages], names)

    validate_status(status, expected_statuses=EXPECTED_STATUSES)

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


def _verify_func(ys_runner, ys_expected):
    import numpy as np

    for y_runner, y_expected in zip(ys_runner, ys_expected):
        if not all(np.allclose(a, b) for a, b in zip(y_runner.values(), y_expected.values())):
            return False
    return True


if __name__ == "__main__":
    main()
