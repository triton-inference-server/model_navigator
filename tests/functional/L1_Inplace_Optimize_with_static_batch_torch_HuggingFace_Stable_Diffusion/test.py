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
    status.format(name=name, ind=0) for status in EXPECTED_STATUSES_TEMPLATE for name in ("clip", "unet", "vae")
]


def get_pipeline():

    # pytype: disable=import-error
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    import model_navigator as nav

    # pytype: enable=import-error

    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    optimize_config = nav.OptimizeConfig(
        batching=False,  # because timestep input to the unet is not batched
        verbose=True,
        runners=(
            "OnnxCUDA",
            "OnnxTensorRT",
            "TorchCUDA",
            "TorchScriptCUDA",
            "TensorRT",
        ),
    )

    def clip_output_mapping(output):
        return BaseModelOutputWithPooling(**output)

    pipe.text_encoder = nav.Module(
        pipe.text_encoder,
        name="clip",
        optimize_config=optimize_config,
        output_mapping=clip_output_mapping,
    )
    pipe.unet = nav.Module(
        pipe.unet,
        name="unet",
        optimize_config=optimize_config,
    )
    pipe.vae.decoder = nav.Module(
        pipe.vae.decoder,
        name="vae",
        optimize_config=optimize_config,
    )

    return pipe


def get_dataloader():
    return [
        ["a photo of an astronaut riding a horse on mars"],  # batch size 1
        ["dog", "cat"],  # batch size 2
    ]


def main():
    import torch  # pytype: disable=import-error
    import transformers  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.inplace.registry import module_registry
    from tests import utils
    from tests.functional.common.utils import collect_optimize_statuses, validate_status

    transformers.modeling_utils.get_parameter_device = lambda parameter: torch.device("cuda")

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

    pipe = get_pipeline()
    dataloader = get_dataloader()

    for batch in dataloader:
        pipe(batch)

    nav.optimize()

    for batch in dataloader:
        pipe(batch)

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
