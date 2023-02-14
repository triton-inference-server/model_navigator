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
"""e2e tests for exporting BERT TensorFlow model from Deep Learning Examples"""
import argparse
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/tensorflow:{version}-py3",
    "repository": "https://github.com/NVIDIA/DeepLearningExamples",
    "model_dir": "TensorFlow2/LanguageModeling/BERT/",
}


def main():
    import model_navigator as nav
    from tests import utils
    from tests.functional.common.tests.dle_bert_tf import dle_bert_tf
    from tests.functional.common.utils import collect_status

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

    package = dle_bert_tf(
        model_dir=METADATA["model_dir"],
        git_url=METADATA["repository"],
        max_batch_size=16,
        input_names=(
            "input_word_ids",
            "input_mask",
            "input_type_ids",
        ),
        trt_profile=nav.TensorRTProfile()
        .add("input_word_ids", (1, 128), (8, 128), (16, 128))
        .add("input_mask", (1, 128), (8, 128), (16, 128))
        .add("input_type_ids", (1, 128), (8, 128), (16, 128)),
    )

    status_file = args.status
    status = collect_status(package.status)
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
