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
import copy
import pathlib
import tempfile

import yaml
from loguru import logger

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_PACKAGES = 3
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
    for name, range_ind in (("encoder", 1), ("decoder", 2))
    for ind in range(range_ind)
]

DEVICE = "cuda"
MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 8


# Whisper is deleting samples, always return copy for inference
class CopyList(list):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        return copy.deepcopy(item)

    def __iter__(self):
        for item in super().__iter__():
            yield copy.deepcopy(item)


class CopyDict(dict):
    def __getitem__(self, key):
        item = super().__getitem__(key)
        return copy.deepcopy(item)

    def items(self):
        for key, value in super().items():
            yield copy.deepcopy(key), copy.deepcopy(value)

    def __iter__(self):
        return iter(self.items())


def get_pipeline():
    # pytype: disable=import-error
    from transformers import pipeline
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

    import model_navigator as nav

    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=DEVICE,
    )
    # TODO: Remove WAR when fixed
    # WAR: Disable module optimization
    # pipe.model.proj_out = nav.Module(
    #     pipe.model.proj_out,
    #     name="proj_out",
    # )
    pipe.model.model.encoder = nav.Module(
        pipe.model.model.encoder,
        name="encoder",
        output_mapping=lambda x: BaseModelOutput(**x),
    )
    pipe.model.model.decoder = nav.Module(
        pipe.model.model.decoder,
        name="decoder",
        output_mapping=lambda x: BaseModelOutputWithPastAndCrossAttentions(**x),
    )

    return pipe


def get_dataloader():
    from datasets import load_dataset  # pytype: disable=import-error

    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    dataloader = CopyList()
    for i in range(5):
        dataloader.append((BATCH_SIZE, CopyDict(**{"inputs": dataset[i]["audio"]})))

    return dataloader


def get_config():
    import model_navigator as nav

    _default_runners = set()
    for name, runner in nav.runners.registry.runner_registry.items():
        if runner.is_default and not runner.is_experimental:
            _default_runners.add(name)

    # pytype: enable=import-error
    optimize_config = nav.OptimizeConfig(
        verbose=True,
        optimization_profile=nav.OptimizationProfile(max_batch_size=BATCH_SIZE),
        runners=tuple(_default_runners),
    )

    return optimize_config


def main():
    import torch  # pytype: disable=import-error
    import transformers  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.inplace.registry import module_registry
    from tests.functional.common.utils import collect_optimize_statuses, validate_status

    transformers.modeling_utils.get_parameter_device = lambda parameter: torch.device(DEVICE)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    args = parser.parse_args()

    logger.debug(f"CLI args: {args}")

    logger.info("Loading model...")
    pipe = get_pipeline()
    logger.info("Model loaded")

    logger.info("Loading dataset...")
    dataloader = get_dataloader()
    logger.info(f"Dataset loaded. Dataset length: {len(dataloader)}")

    config = get_config()

    optimize_status = nav.optimize(pipe, dataloader=dataloader, config=config)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = pathlib.Path(tmpdir) / "optimized_status.yaml"
        optimize_status.to_file(tmp_file)

    names, packages = [], []
    for name, module in module_registry.items():
        for i, package in enumerate(module.wrapper.packages):
            names.append(f"{name}.{i}")
            packages.append(package)
    assert (
        len(packages) == EXPECTED_PACKAGES
    ), f"Wrong number of packages. Got {len(packages)}. Expected: {EXPECTED_PACKAGES}"

    status = collect_optimize_statuses([package.status for package in packages], names)

    # Profile
    profile_status = nav.profile(pipe, dataloader, device=DEVICE, verbose=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = pathlib.Path(tmpdir) / "profiling_results.yaml"
        profile_status.to_file(tmp_file)

    validate_status(status, expected_statuses=EXPECTED_STATUSES)

    status_file = args.status
    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
