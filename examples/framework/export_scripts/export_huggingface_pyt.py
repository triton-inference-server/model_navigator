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

import model_navigator as nav

EXPORT_CONFIGS = {
    "distilbert-base-uncased": {
        "model_name": "distilbert-base-uncased",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
    },
    "gpt2": {
        "model_name": "gpt2",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
    },
    "bert-base-uncased": {
        "model_name": "bert-base-uncased",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
    },
    "distilgpt2": {
        "model_name": "distilgpt2",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
    },
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
        "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
    },
    "bert-base-chinese": {
        "model_name": "bert-base-chinese",
        "dataset_name": "imdb",
        "padding": "max_length",
        "max_sequence_len": 384,
        "max_bs": 2,
        "sample_count": 10,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", type=str, choices=list(EXPORT_CONFIGS.keys()))
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
        print(list(EXPORT_CONFIGS.keys()))
        return
    export_config = EXPORT_CONFIGS[args.model_name]
    # pytype: disable=not-callable # TODO why is not-calleble being raised by pytype?
    pkg_desc = nav.contrib.huggingface.torch.export(**export_config, override_workdir=True)
    # pytype: enable=not-callable
    output_path = args.output_path or f"{args.model_name}_pyt.nav"
    pkg_desc.save(output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
