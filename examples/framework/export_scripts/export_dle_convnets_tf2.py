#!/usr/bin/env python3
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import subprocess
import sys
import tempfile

import tensorflow as tf  # pytype: disable=import-error

import model_navigator as nav

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

DATALOADER = [tf.random.uniform(shape=[2, 224, 224, 3], minval=0, maxval=1, dtype=tf.dtypes.float32)]

MODEL_NAMES = ["EfficientNet-v1-B0", "EfficientNet-v1-B4", "EfficientNet-v2-S"]


def setup_env(workdir):
    cmd = ["git", "clone", "https://github.com/NVIDIA/DeepLearningExamples", f"{workdir}/DeepLearningExamples"]
    subprocess.run(cmd, check=True)
    sys.path.append(f"{workdir}/DeepLearningExamples/TensorFlow2/Classification/ConvNets/")


def get_verification_status_dummy(runner):
    """Dummy verification function."""
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", type=str, choices=MODEL_NAMES)
    group.add_argument(
        "--list-models",
        action="store_true",
    )
    parser.add_argument(
        "--output-path",
        type=str,
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    if args.list_models:
        print(MODEL_NAMES)
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_env(tmpdir)
        from config.defaults import Config, base_config  # pytype: disable=import-error

        if args.model_name == "EfficientNet-v1-B0":
            from config.efficientnet_v1.b0_cfg import config as model_config  # pytype: disable=import-error
            from model.efficientnet_model_v1 import Model  # pytype: disable=import-error
        elif args.model_name == "EfficientNet-v1-B4":
            from config.efficientnet_v1.b4_cfg import config as model_config  # pytype: disable=import-error
            from model.efficientnet_model_v1 import Model  # pytype: disable=import-error
        elif args.model_name == "EfficientNet-v2-S":
            from config.efficientnet_v2.s_cfg import config as model_config  # pytype: disable=import-error
            from model.efficientnet_model_v2 import Model  # pytype: disable=import-error
        else:
            raise ValueError(f"Unknown model: {args.model_name}")

        config = Config(**{**base_config.train, **base_config.runtime, **base_config.data, **base_config.predict})
        config.mparams = Config(model_config)
        config.num_classes = config.mparams.num_classes
        config.train_batch_size = config.batch_size
        config.mode = "predict"

        model = Model(config)

        pkg_desc = nav.tensorflow.export(
            model=model,
            model_name=f"{args.model_name}_pyt",
            dataloader=DATALOADER,
            opset=13,
            override_workdir=True,
        )

        for model_status in pkg_desc.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                if runtime_results.status == nav.Status.OK:
                    runner = pkg_desc.get_runner(
                        format=model_status.format,
                        precision=model_status.precision,
                        runtime=runtime_results.runtime,
                    )
                    verified = get_verification_status_dummy(runner)
                    if verified:
                        pkg_desc.set_verified(
                            format=model_status.format,
                            precision=model_status.precision,
                            runtime=runtime_results.runtime,
                        )
                        nav.LOGGER.info(
                            f"{model_status.format=}, {model_status.precision=}, {runtime_results.runtime=} verified."
                        )
                    else:
                        nav.LOGGER.warning(
                            f"{model_status.format=}, {model_status.precision=}, {runtime_results.runtime=} not verified."
                        )

        output_path = args.output_path or f"{args.model_name}_tf2.nav"
        pkg_desc.save(output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
