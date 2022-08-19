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

import tensorflow  # pytype: disable=import-error

import model_navigator as nav

EXPORT_CONFIGS = {
    "distilbert-base-uncased": {
        "model_name": "distilbert-base-uncased",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
        "expected_runtimes": {
            "tf-savedmodel": [nav.RuntimeProvider.TF.value],
            "trt-fp32": [nav.RuntimeProvider.TRT.value],
            "onnx": [nav.RuntimeProvider.CUDA.value],
            "tf-trt-fp32": [nav.RuntimeProvider.TF.value],
        },
    },
    "distilgpt2": {
        "model_name": "distilgpt2",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
        "expected_runtimes": {
            "tf-savedmodel": [nav.RuntimeProvider.TF.value],
            "trt-fp32": [nav.RuntimeProvider.TRT.value],
            "onnx": [nav.RuntimeProvider.CUDA.value],
            "tf-trt-fp32": [nav.RuntimeProvider.TF.value],
        },
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str, choices=list(EXPORT_CONFIGS.keys()))
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    export_config = EXPORT_CONFIGS[args.model_name]

    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    nav_workdir = Path(args.workdir)
    nav.LOGGER.info(f"Testing {export_config['model_name']}...")
    expected_runtimes = export_config.pop("expected_runtimes")
    # pytype: disable=not-callable # TODO why is not-calleble being raised by pytype?
    pkg_desc = nav.contrib.huggingface.tensorflow.export(
        workdir=nav_workdir,
        **export_config,
        target_precisions=(nav.TensorRTPrecision.FP32,),
        opset=13,
        profiler_config=nav.ProfilerConfig(measurement_request_count=20),
        runtimes=(nav.RuntimeProvider.TF, nav.RuntimeProvider.CUDA, nav.RuntimeProvider.TRT),
    )
    # pytype: enable=not-callable
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

    nav.save(pkg_desc, Path(args.workdir) / f"{export_config['model_name']}_tf2.nav")
    nav.LOGGER.info(f"{export_config['model_name']} passed.")
