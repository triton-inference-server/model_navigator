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
import tensorflow  # pytype: disable=import-error
from transformers import FlaxT5Model, T5Tokenizer  # pytype: disable=import-error

import model_navigator as nav

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)


def get_model():
    model = FlaxT5Model.from_pretrained("t5-small")

    def call_wrapper(input_ids, decoder_input_ids, params):
        return model.__call__(input_ids=input_ids, decoder_input_ids=decoder_input_ids, params=params)

    return call_wrapper, model._params


def get_dataloader():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="np").input_ids
    decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids
    dataloader = [{"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}]

    return dataloader


def get_verify_function():
    """Define verify function that compares outputs of the torch model and the optimized model."""

    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                [np.allclose(a, b, rtol=1.0e-3, atol=1.0e-3) for a, b in zip(y_runner.values(), y_expected.values())]
            ):
                return False
        return True

    return verify_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="gpt2.nav",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model, params = get_model()
    dataloader = get_dataloader()
    verify_func = get_verify_function()

    package = nav.jax.optimize(
        model=model,
        model_params=params,
        dataloader=dataloader,
        verify_func=verify_func,
    )

    nav.package.save(package, args.output_path, override=True)


if __name__ == "__main__":
    main()
