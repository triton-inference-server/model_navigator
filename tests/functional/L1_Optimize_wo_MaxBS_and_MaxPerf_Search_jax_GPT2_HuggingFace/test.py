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
"""e2e tests for exporting GPT2 JAX model from HuggingFace"""
import argparse
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/tensorflow:{version}-tf2-py3",
}


def main():
    import tensorflow as tf  # pytype: disable=import-error
    from transformers import FlaxGPT2Model, GPT2Tokenizer  # pytype: disable=import-error

    import model_navigator as nav
    from tests import utils
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

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = FlaxGPT2Model.from_pretrained("gpt2")

    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="np")
    dataloader = [encoded_input]
    trt_profile = (
        nav.TensorRTProfile()
        .add("input_ids", (1, 64), (2, 64), (4, 64))
        .add("attention_mask", (1, 64), (2, 64), (4, 64))
    )

    LOGGER.info("Testing GPT2")
    package = nav.jax.optimize(
        model=model.__call__,
        model_params=model._params,
        dataloader=dataloader,
        verbose=True,
        custom_configs=(
            nav.TensorRTConfig(trt_profile=trt_profile),
            nav.TensorFlowTensorRTConfig(trt_profile=trt_profile),
        ),
        profiler_config=nav.ProfilerConfig(batch_sizes=[1, 2, 4]),
    )

    status_file = args.status
    status = collect_status(package.status)
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
