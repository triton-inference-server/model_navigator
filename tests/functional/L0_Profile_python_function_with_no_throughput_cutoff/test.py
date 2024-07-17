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

import yaml
from loguru import logger

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}


def main():
    import model_navigator as nav

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )

    args = parser.parse_args()

    logger.debug(f"CLI args: {args}")

    def call(input_):
        return input_

    dataloader = [(1, "This is example input"), (1, "This is example input")]

    profiling_status_path = pathlib.Path("profile_status.yaml")
    profile_status = nav.profile(call, dataloader, throughput_cutoff_threshold=None)
    profile_status.to_file(profiling_status_path)

    detailed_results = None

    try:
        with open(profiling_status_path) as file:
            profile_status = yaml.safe_load(file.read())

        eager_results = (
            profile_status.get("profiling_results")
            .get("models", {})
            .get("python", {})
            .get("runners", {})
            .get("eager", {})
        )

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
