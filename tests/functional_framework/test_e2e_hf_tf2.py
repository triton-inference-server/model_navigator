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
import tempfile
from pathlib import Path

import model_navigator as nav

EXPORT_CONFIGS = {
    "distilbert-base-uncased": {
        "model_name": "distilbert-base-uncased",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
        "expected_formats": ("tf-savedmodel", "tf-trt-fp32"),
    },
    "distilgpt2": {
        "model_name": "distilgpt2",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
        "expected_formats": ("tf-savedmodel", "tf-trt-fp32"),
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str, choices=list(EXPORT_CONFIGS.keys()))
    args = parser.parse_args()
    export_config = EXPORT_CONFIGS[args.model_name]
    with tempfile.TemporaryDirectory() as tmp_dir:
        nav_workdir = Path(tmp_dir) / "navigator_workdir"
        nav.LOGGER.info(f"Testing {export_config['model_name']}...")
        expected_formats = export_config.pop("expected_formats")
        # pytype: disable=not-callable # TODO why is not-calleble being raised by pytype?
        pkg_desc = nav.contrib.huggingface.tensorflow.export(
            workdir=nav_workdir, **export_config, target_precisions=(nav.TensorRTPrecision.FP32,)
        )
        # pytype: enable=not-callable
        for format, status in pkg_desc.get_formats_status().items():
            status = list(status.values())[0]
            assert (status == nav.Status.OK) == (
                format in expected_formats
            ), f"{format} status is {status.value}, but expected formats are {expected_formats}."
        nav.LOGGER.info(f"{export_config['model_name']} passed.")
    nav.LOGGER.info("All models passed.")
