#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""e2e tests for exporting Distilled GTP2 TensorFlow model from HuggingFace"""

import argparse
import itertools
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/tensorflow:{version}-tf2-py3",
}


def main():
    import tensorflow  # pytype: disable=import-error
    from datasets import load_dataset  # pytype: disable=import-error
    from transformers import AutoTokenizer, TensorType  # pytype: disable=import-error
    from transformers.models.gpt2.configuration_gpt2 import GPT2OnnxConfig  # pytype: disable=import-error
    from transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2LMHeadModel  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.frameworks import Framework
    from tests import utils
    from tests.functional.common import huggingface_utils
    from tests.functional.common.utils import collect_optimize_status

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Timeout for test.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=utils.DEFAULT_LOG_FORMAT)
    LOGGER.debug(f"CLI args: {args}")

    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    model_name = "distilgpt2"
    dataset_name = "imdb"

    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    model.config.return_dict = True
    model.config.use_cache = False
    max_batch_size = 4

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    max_sequence_length = getattr(tokenizer, "model_max_length", None)

    onnx_config = GPT2OnnxConfig(model.config)
    input_names = tuple(onnx_config.inputs.keys())
    output_names = tuple(onnx_config.outputs.keys())
    dynamic_axes = {
        name: axes for name, axes in itertools.chain(onnx_config.inputs.items(), onnx_config.outputs.items())
    }
    opset = onnx_config.default_onnx_opset

    dataset = load_dataset(dataset_name)["train"]
    dataloader_factory = huggingface_utils.HFDataLoaderFactory(
        dataset=dataset,
        tokenizer=tokenizer,
        preprocess_function=huggingface_utils.get_default_preprocess_function(
            dataset_name, tokenizer, max_sequence_length
        ),
        inputs=input_names,
        padding="max_length",
        max_sequence_length=max_sequence_length,
        return_tensors=TensorType.TENSORFLOW,
    )
    dataloader = dataloader_factory(max_batch_size, framework=Framework.TENSORFLOW)

    package = nav.tensorflow.optimize(
        model=model,
        dataloader=dataloader,
        sample_count=10,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        custom_configs=(
            nav.OnnxConfig(
                opset=opset,
                dynamic_axes=dynamic_axes,
            ),
        ),
    )

    status_file = args.status
    status = collect_optimize_status(package.status)
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
