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
    "max_batch_size",
]


def main():
    import numpy as np
    import torch  # pytype: disable=import-error

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

    model = torch.nn.Identity()
    dataloader = [np.random.rand(2, 5).astype("float32") for _ in range(5)]

    max_batch_size = nav.utilities.find_max_batch_size_till_oom(
        nav.Framework.TORCH,
        model,
        dataloader,
        max_batch_size_search_limit=2048,
    )

    status = {}
    if max_batch_size == 2048:
        status["max_batch_size"] = "OK"
    else:
        status["max_batch_size"] = "FAIL"

    status_file = args.status
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)


if __name__ == "__main__":
    main()
