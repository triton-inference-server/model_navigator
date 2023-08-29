#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""e2e tests for exporting PyTorch identity model"""
import argparse
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_PACKAGES = 4
EXPECTED_STATUSES_TEMPLATE = [
    "{name}.{ind}.onnx.OnnxCUDA",
    "{name}.{ind}.onnx.OnnxTensorRT",
    "{name}.{ind}.torch.TorchCUDA",
    "{name}.{ind}.torchscript-script.TorchScriptCUDA",
    "{name}.{ind}.torchscript-trace.TorchScriptCUDA",
    "{name}.{ind}.trt-fp16.TensorRT",
    "{name}.{ind}.trt-fp32.TensorRT",
]
EXPECTED_STATUSES = [
    status.format(name=name, ind=ind)
    for status in EXPECTED_STATUSES_TEMPLATE
    for name, range_ind in (("encoder", 1), ("decoder", 2), ("proj_out", 1))
    for ind in range(range_ind)
]

DEVICE = "cuda"
MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 8


def get_pipeline():
    # pytype: disable=import-error
    from transformers import pipeline
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

    import model_navigator as nav

    # pytype: enable=import-error
    optimize_config = nav.OptimizeConfig(
        runners=(
            "OnnxCUDA",
            "OnnxTensorRT",
            "TorchCUDA",
            "TorchScriptCUDA",
            "TensorRT",
        ),
        verbose=True,
        optimization_profile=nav.OptimizationProfile(max_batch_size=BATCH_SIZE),
    )
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=DEVICE,
    )
    pipe.model.model.encoder = nav.Module(
        pipe.model.model.encoder,
        name="encoder",
        optimize_config=optimize_config,
        output_mapping=lambda x: BaseModelOutput(**x),
    )
    pipe.model.model.decoder = nav.Module(
        pipe.model.model.decoder,
        name="decoder",
        optimize_config=optimize_config,
        output_mapping=lambda x: BaseModelOutputWithPastAndCrossAttentions(**x),
    )
    pipe.model.proj_out = nav.Module(
        pipe.model.proj_out,
        name="proj_out",
        optimize_config=optimize_config,
    )
    return pipe


def main():
    import torch  # pytype: disable=import-error
    import transformers  # pytype: disable=import-error
    from datasets import load_dataset  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.inplace.registry import module_registry
    from tests import utils
    from tests.functional.common.utils import collect_optimize_statuses, validate_status

    transformers.modeling_utils.get_parameter_device = lambda parameter: torch.device(DEVICE)

    nav.inplace_config.mode = nav.Mode.RECORDING

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

    LOGGER.info("Loading model...")
    pipe = get_pipeline()
    LOGGER.info("Model loaded")
    LOGGER.info("Loading dataset...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    LOGGER.info(f"Dataset loaded. Dataset length: {len(dataset)}")

    for batch in dataset:
        sample = batch["audio"]
        pipe(sample.copy(), batch_size=BATCH_SIZE)

    # additional inference to capture max sequence length output
    pipe(dataset[0]["audio"].copy(), batch_size=BATCH_SIZE, generate_kwargs={"min_new_tokens": 512})

    nav.optimize()

    for batch in dataset:
        sample = batch["audio"]
        pipe(sample.copy(), batch_size=BATCH_SIZE)

    names, packages = [], []
    for name, module in module_registry.items():
        for i, package in enumerate(getattr(module._wrapper, "_packages", [])):
            names.append(f"{name}.{i}")
            packages.append(package)
    assert len(packages) == EXPECTED_PACKAGES, "Wrong number of packages."

    status_file = args.status
    status = collect_optimize_statuses([package.status for package in packages], names)

    validate_status(status, expected_statuses=EXPECTED_STATUSES)

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
