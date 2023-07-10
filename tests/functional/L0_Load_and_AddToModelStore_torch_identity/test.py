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
import tempfile

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
]


def main():
    import model_navigator as nav
    from tests import utils
    from tests.functional.common.utils import collect_optimize_status, validate_model_repository, validate_status
    from tests.utils import get_assets_path

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

    package_path = get_assets_path() / "packages" / "torch_identity.nav"
    package = nav.package.load(package_path)

    status_file = args.status
    status = collect_optimize_status(package.status)

    validate_status(status, expected_statuses=EXPECTED_STATUES)

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")

    with tempfile.TemporaryDirectory() as tempdir:
        model_repository = pathlib.Path(tempdir) / "model_repository"
        model_repository.mkdir()

        nav.triton.model_repository.add_model_from_package(
            model_repository_path=model_repository,
            model_name="Identity",
            package=package,
        )
        LOGGER.info(f"Create deployment in {model_repository}")

        validate_model_repository(model_repository=model_repository, model_name="Identity")


if __name__ == "__main__":
    main()
