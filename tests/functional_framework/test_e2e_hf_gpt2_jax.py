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

from transformers import FlaxGPT2Model, GPT2Tokenizer

import model_navigator as nav

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    nav_workdir = Path(args.workdir)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = FlaxGPT2Model.from_pretrained("gpt2")

    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="np")
    dataloader = [encoded_input]

    nav.LOGGER.info("Testing GPT2")
    expected_formats = (nav.Format.TF_SAVEDMODEL,)
    pkg_desc = nav.jax.export(
        model=model.__call__, model_params=model._params, dataloader=dataloader, workdir=nav_workdir
    )

    for format, runtimes_status in pkg_desc.get_formats_status().items():
        for runtime, status in runtimes_status.items():
            assert (status == nav.Status.OK) == (
                format in expected_formats
            ), f"{format} {runtime} status is {status}, but expected formats are {expected_formats}."
    nav.LOGGER.info("GPT2 passed.")
    nav.save(pkg_desc, Path(args.workdir) / "gpt2_jax.nav")
