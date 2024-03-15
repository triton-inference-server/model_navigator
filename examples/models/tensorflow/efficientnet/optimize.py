#!/usr/bin/env python3
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

import argparse

import numpy as np
import tensorflow as tf  # pytype: disable=import-error

import model_navigator as nav

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


MODEL_NAMES = ["EfficientNet-v1-B0", "EfficientNet-v1-B4", "EfficientNet-v2-S"]


def get_model(model_name: str):
    from config.defaults import Config, base_config  # pytype: disable=import-error

    if model_name == "EfficientNet-v1-B0":
        from config.efficientnet_v1.b0_cfg import config as model_config  # pytype: disable=import-error
        from model.efficientnet_model_v1 import Model  # pytype: disable=import-error
    elif model_name == "EfficientNet-v1-B4":
        from config.efficientnet_v1.b4_cfg import config as model_config  # pytype: disable=import-error
        from model.efficientnet_model_v1 import Model  # pytype: disable=import-error
    elif model_name == "EfficientNet-v2-S":
        from config.efficientnet_v2.s_cfg import config as model_config  # pytype: disable=import-error
        from model.efficientnet_model_v2 import Model  # pytype: disable=import-error
    else:
        raise ValueError(f"Unknown model: {model_name}")

    config = Config(**{**base_config.train, **base_config.runtime, **base_config.data, **base_config.predict})
    config.mparams = Config(model_config)
    config.num_classes = config.mparams.num_classes
    config.train_batch_size = config.batch_size
    config.mode = "predict"

    model = Model(config)

    return model


def get_dataloader():
    return [tf.random.uniform(shape=[2, 224, 224, 3], minval=0, maxval=1, dtype=tf.dtypes.float32)]


def get_verify_function():
    def verify_func(ys_runner, ys_expected):
        """Verify that at least 99% max probability tokens match on any given batch."""
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.mean(a.argmax(axis=1) == b.argmax(axis=1)) > 0.99
                for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        default="efficientnet.nav",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = get_model(args.model_name)
    dataloader = get_dataloader()
    verify_func = get_verify_function()

    package = nav.tensorflow.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_func,
    )

    nav.package.save(package, args.output_path, override=True)


if __name__ == "__main__":
    main()
