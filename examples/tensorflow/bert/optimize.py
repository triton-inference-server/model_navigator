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
import itertools

import numpy as np
from datasets import load_dataset  # pytype: disable=import-error
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TensorType  # pytype: disable=import-error
from transformers.models.distilbert.configuration_distilbert import DistilBertOnnxConfig  # pytype: disable=import-error
from transformers.models.distilbert.modeling_tf_distilbert import (  # pytype: disable=import-error
    TFDistilBertForMaskedLM,
)

import model_navigator as nav


def get_model(model_name: str):
    model = TFDistilBertForMaskedLM.from_pretrained(model_name)
    model.config.return_dict = True
    model.config.use_cache = False
    return model


def get_dataloader(model_name: str, dataset_name: str, max_batch_size: int, num_samples: int):

    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_sequence_length = getattr(tokenizer, "model_max_length", None)

    onnx_config = DistilBertOnnxConfig(model_config)
    input_names = tuple(onnx_config.inputs.keys())
    dataset = load_dataset(dataset_name)["train"]

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_sequence_length)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset.column_names if c not in input_names]
    )
    dataloader = tokenized_dataset.to_tf_dataset(
        columns=tokenized_dataset.column_names,
        shuffle=True,
        batch_size=max_batch_size,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True, max_length=max_sequence_length, return_tensors=TensorType.TENSORFLOW
        ),
    )

    return [sample for sample, _ in zip(dataloader, range(num_samples))]


def get_verify_function():
    def verify_func(ys_runner, ys_expected):
        """Verify that at least 99% max probability tokens match on any given batch."""
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.mean(a.argmax(axis=2) == b.argmax(axis=2)) > 0.99
                for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def get_configuration(model_name: str):

    model_config = AutoConfig.from_pretrained(model_name)
    onnx_config = DistilBertOnnxConfig(model_config)
    input_names = tuple(onnx_config.inputs.keys())
    output_names = tuple(onnx_config.outputs.keys())
    dynamic_axes = {
        name: axes for name, axes in itertools.chain(onnx_config.inputs.items(), onnx_config.outputs.items())
    }
    opset = onnx_config.default_onnx_opset

    configuration = {
        "input_names": input_names,
        "output_names": output_names,
        "sample_count": 10,
        "verbose": True,
        "custom_configs": [nav.OnnxConfig(opset=opset, dynamic_axes=dynamic_axes)],
    }
    return configuration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="distilbert.nav",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = "distilbert-base-uncased"
    dataset_name = "imdb"
    max_batch_size = 4
    num_samples = 100

    model = get_model(model_name)
    dataloader = get_dataloader(
        model_name=model_name,
        dataset_name=dataset_name,
        max_batch_size=max_batch_size,
        num_samples=num_samples,
    )
    verify_func = get_verify_function()
    configuration = get_configuration(model_name)

    package = nav.tensorflow.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_func,
        **configuration,
    )

    nav.package.save(package, args.output_path, override=True)


if __name__ == "__main__":
    main()
