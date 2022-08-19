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
from pathlib import Path

import tensorflow as tf  # pytype: disable=import-error
from config.defaults import Config, base_config  # pytype: disable=import-error

import model_navigator as nav

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

DATALOADER = [tf.random.uniform(shape=[2, 224, 224, 3], minval=0, maxval=1, dtype=tf.dtypes.float32)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=["EfficientNet-v1-B0", "EfficientNet-v1-B4", "EfficientNet-v2-S"],
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    model_name = args.model_name
    nav.LOGGER.info(f"Testing {model_name}...")
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

    nav_workdir = Path(args.workdir)

    pkg_desc = nav.tensorflow.export(
        model=model,
        model_name=model_name.lower() + "_tf2",
        workdir=nav_workdir,
        dataloader=DATALOADER,
        target_precisions=(nav.TensorRTPrecision.FP32,),
        opset=13,
        override_workdir=True,
        profiler_config=nav.ProfilerConfig(measurement_request_count=20),
    )
    expected_runtimes = {
        "tf-savedmodel": [nav.RuntimeProvider.TF.value],
        "onnx": [nav.RuntimeProvider.CPU.value, nav.RuntimeProvider.CUDA.value],
        "trt-fp32": [nav.RuntimeProvider.TRT.value],
        "tf-trt-fp32": [nav.RuntimeProvider.TF.value],
    }
    nav.LOGGER.info(f"{pkg_desc.get_formats_status()=}")
    for format, runtimes_status in pkg_desc.get_formats_status().items():
        for runtime, status in runtimes_status.items():
            if runtime in expected_runtimes.get(format, {}):
                assert (
                    status == nav.Status.OK
                ), f"{format} {runtime} status is {status}, but expected runtimes are {expected_runtimes}."
            else:
                if status == nav.Status.OK:
                    nav.LOGGER.warning(f"{format} {runtime} status is {status} but it is not in expected runtimes.")
    nav.save(pkg_desc, nav_workdir / f"{model_name}_tf2.nav")

    nav.LOGGER.info(f"{model_name} passed.")
