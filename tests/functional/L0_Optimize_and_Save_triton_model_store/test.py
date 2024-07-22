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
"""e2e tests for exporting PyTorch identity model with Triton model store"""

import argparse
import pathlib
import tempfile

import yaml
from loguru import logger

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_STATUES = [
    "onnx.OnnxCUDA",
    "trt-fp16.TensorRT",
]


def main():
    import numpy as np
    import torch  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.configuration import SelectedRuntimeStrategy
    from model_navigator.exceptions import ModelNavigatorUserInputError
    from tests.functional.common.utils import collect_optimize_status, validate_status

    nav.inplace_config.min_num_samples = 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    args = parser.parse_args()

    logger.debug(f"CLI args: {args}")

    model = torch.nn.Identity()
    dataloader = [(1, torch.randn(2, 3)) for _ in range(2)]

    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(np.allclose(a, b) for a, b in zip(y_runner.values(), y_expected.values())):
                return False
        return True

    optimize_config = nav.OptimizeConfig(
        verbose=True,
        optimization_profile=nav.OptimizationProfile(batch_sizes=[1, 8, 16], stability_percentage=100),
        verify_func=verify_func,
        runners=(
            "OnnxCUDA",
            "TensorRT",
        ),
        custom_configs=[
            nav.TensorRTConfig(
                precision=(nav.TensorRTPrecision.FP16,),
            ),
        ],
    )
    model = nav.Module(model, name="identity")

    optimize_status = nav.optimize(func=model, dataloader=dataloader, config=optimize_config)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = pathlib.Path(tmpdir) / "optimized_status.yaml"
        optimize_status.to_file(tmp_file)

    packages = getattr(model._wrapper, "_packages", [])
    assert len(packages) == 1, "Package is not created."
    package = packages[0]

    status_file = args.status
    status = collect_optimize_status(package.status)

    validate_status(status, expected_statuses=EXPECTED_STATUES)

    nav.load_optimized()

    # Generate model store
    with tempfile.TemporaryDirectory() as tmp:
        model_store = pathlib.Path(tmp) / "model_store"

        model_repo = model_store / "best"
        model_repo.mkdir(exist_ok=True, parents=True)
        model.triton_model_store(model_repository_path=model_repo)

        model_repo = model_store / "trt"
        model_repo.mkdir(exist_ok=True, parents=True)
        model.triton_model_store(
            model_repository_path=model_repo,
            strategy=SelectedRuntimeStrategy("trt-fp16", "TensorRT"),
        )

        model_repo = model_store / "onnx"
        model_repo.mkdir(exist_ok=True, parents=True)
        model.triton_model_store(
            model_repository_path=model_repo,
            strategy=SelectedRuntimeStrategy("onnx", "OnnxCUDA"),
            model_name="identity2",
        )
        try:
            model.triton_model_store(
                model_repository_path=model_repo,
                strategy=SelectedRuntimeStrategy("onnx", "OnnxCUDA"),
                package_idx=100,
            )
            raise AssertionError("Expected ModelNavigatorUserInputError")
        except ModelNavigatorUserInputError as e:
            assert e.message.startswith("Incorrect package index package_idx=")

        # displayed stored models for debugging
        for child in model_store.glob("**/*.*"):
            logger.info(f"{child}")

        assert (model_store / "best/identity/config.pbtxt").exists()
        assert (model_store / "onnx/identity2/1/model.onnx").exists()
        assert (model_store / "trt/identity/1/model.plan").exists()

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
