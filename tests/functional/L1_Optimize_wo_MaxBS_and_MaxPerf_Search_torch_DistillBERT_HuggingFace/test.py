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
"""e2e tests for exporting DistillBERT PyTorch model from HuggingFace"""

import argparse
import itertools
import pathlib

import yaml
from loguru import logger

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}


def main():
    from datasets import load_dataset  # pytype: disable=import-error
    from transformers import AutoTokenizer, TensorType  # pytype: disable=import-error
    from transformers.models.distilbert.configuration_distilbert import (  # pytype: disable=import-error
        DistilBertOnnxConfig,
    )
    from transformers.models.distilbert.modeling_distilbert import DistilBertForMaskedLM  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.frameworks import Framework
    from tests.functional.common import huggingface_utils
    from tests.functional.common.utils import collect_optimize_status

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    args = parser.parse_args()

    logger.debug(f"CLI args: {args}")

    model_name = "distilbert-base-uncased"
    dataset_name = "imdb"

    model = DistilBertForMaskedLM.from_pretrained(model_name)
    model.config.return_dict = True
    max_batch_size = 4

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_sequence_length = min(128, getattr(tokenizer, "model_max_length", 128))

    onnx_config = DistilBertOnnxConfig(model.config)
    input_names = tuple(onnx_config.inputs.keys())
    output_names = tuple(onnx_config.outputs.keys())
    dynamic_axes = {
        name: axes for name, axes in itertools.chain(onnx_config.inputs.items(), onnx_config.outputs.items())
    }

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
        return_tensors=TensorType.PYTORCH,
    )
    dataloader = dataloader_factory(max_batch_size, framework=Framework.TORCH)

    trt_profiles = [
        nav.TensorRTProfile()
        .add("input_ids", (1, max_sequence_length), (8, max_sequence_length), (16, max_sequence_length))
        .add("attention_mask", (1, max_sequence_length), (8, max_sequence_length), (16, max_sequence_length))
    ]

    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        sample_count=10,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        custom_configs=(
            nav.OnnxConfig(
                dynamic_axes=dynamic_axes,
                export_engine=[
                    nav.OnnxTraceExportConfig(),
                    # nav.OnnxDynamoExportConfig(), # TODO: Torch 2.6 works but 2.7 fails
                ],
            ),
            nav.TensorRTConfig(trt_profiles=trt_profiles),
            nav.TorchTensorRTConfig(trt_profiles=trt_profiles),
        ),
        optimization_profile=nav.OptimizationProfile(max_batch_size=16),
    )

    status_file = args.status
    status = collect_optimize_status(package.status)
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
