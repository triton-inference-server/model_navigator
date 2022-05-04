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


def get_verification_status_dummy(runner):
    """Dummy verification function."""
    return False


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

    for model_status in pkg_desc.navigator_status.model_status:
        for runtime_results in model_status.runtime_results:
            if runtime_results.status == nav.Status.OK:
                runner = pkg_desc.get_runner(
                    format=model_status.format,
                    jit_type=model_status.torch_jit,
                    precision=model_status.precision,
                    runtime=runtime_results.runtime,
                )
                verified = get_verification_status_dummy(runner)
                if verified:
                    pkg_desc.set_verified(
                        format=model_status.format,
                        jit_type=model_status.torch_jit,
                        precision=model_status.precision,
                        runtime=runtime_results.runtime,
                    )
                    nav.LOGGER.info(
                        f"{model_status.format=}, {model_status.torch_jit=}, {model_status.precision=}, {runtime_results.runtime=} verified."
                    )
                else:
                    nav.LOGGER.warning(
                        f"{model_status.format=}, {model_status.torch_jit=}, {model_status.precision=}, {runtime_results.runtime=} not verified."
                    )

    output_path = args.output_path or f"{args.model_name}_pyt.nav"
    pkg_desc.save(output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
