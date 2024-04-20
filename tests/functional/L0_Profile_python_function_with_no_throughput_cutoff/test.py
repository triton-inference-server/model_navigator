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
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}


def main():
    import model_navigator as nav
    from tests import utils

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

    def call(input_):
        return input_

    dataloader = [(1, "This is example input"), (1, "This is example input")]

    profiling_status_path = pathlib.Path("profile_status.yaml")
    nav.profile(call, dataloader, status_path=profiling_status_path, throughput_cutoff_threshold=None)

    detailed_results = None

    try:
        with open(profiling_status_path) as file:
            profiling_status = yaml.safe_load(file.read())

        eager_results = profiling_status.get("models", {}).get("python", {}).get("runners", {}).get("eager", {})

        python_eager_status = eager_results.get("status", "FAIL")
        detailed_results = eager_results.get("detailed")

    except Exception:
        python_eager_status = "FAIL"

    assert detailed_results is not None
    assert len(detailed_results) == 2

    status = {"python.eager": python_eager_status}

    status_file = args.status
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)


if __name__ == "__main__":
    main()
