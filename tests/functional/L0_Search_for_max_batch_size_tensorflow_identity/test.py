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
"""e2e tests for exporting TensorFlow identity model"""
import argparse
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/tensorflow:{version}-tf2-py3",
}
EXPECTED_STATUES = [
    "max_batch_size",
]


def main():
    import tensorflow  # pytype: disable=import-error

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

    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    inp = tensorflow.keras.layers.Input((3,))
    layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
    model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
    model = tensorflow.keras.Model(inp, model_output)

    dataloader = [
        tensorflow.random.uniform(shape=[2, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32) for _ in range(2)
    ]

    max_batch_size = nav.utilities.find_max_batch_size_till_oom(
        nav.Framework.TENSORFLOW,
        model,
        dataloader,
        max_batch_size_search_limit=4,
    )

    status = {}
    if max_batch_size == 4:
        status["max_batch_size"] = "OK"
    else:
        status["max_batch_size"] = "FAIL"

    status_file = args.status
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
