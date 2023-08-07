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
"""e2e tests for exporting EfficientNet V1 B0 TensorFlow model from Deep Learning Examples"""
import argparse
import logging
import os
import pathlib
import sys
import tempfile

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/tensorflow:{version}-tf2-py3",
    "repository": "https://github.com/NVIDIA/DeepLearningExamples",
    "model_dir": "TensorFlow2/Classification/ConvNets/",
}


def main():
    from git import Repo

    import model_navigator as nav
    from tests import utils
    from tests.functional.common.tests.dle_convnets_tf import dle_convnets_tf
    from tests.functional.common.utils import collect_optimize_status

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

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = METADATA["model_dir"]
        git_url = METADATA["repository"]

        repo = pathlib.Path(tmp)
        Repo.clone_from(git_url, repo)

        model_dir = repo / model_dir

        sys.path.append(model_dir.as_posix())
        os.chdir(model_dir.as_posix())

        trt_profiles = [
            nav.TensorRTProfile().add(
                "input__0",
                (1, 224, 224, 3),
                (32, 224, 224, 3),
                (64, 224, 224, 3),
            )
        ]

        package = dle_convnets_tf(
            model_name="EfficientNet-v1-B0",
            max_batch_size=64,
            trt_profiles=trt_profiles,
        )
        status_file = args.status
        status = collect_optimize_status(package.status)
        with status_file.open("w") as fp:
            yaml.safe_dump(status, fp)

        LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
