#!/usr/bin/env python3
# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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
import pathlib
import tempfile
from typing import List, Tuple

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

    pipe.text_encoder = nav.Module(
        pipe.text_encoder,
        name="clip",
        output_mapping=lambda output: BaseModelOutputWithPooling(**output),
    )
    pipe.unet = nav.Module(
        pipe.unet,
        name="unet",
    )
    pipe.vae.decoder = nav.Module(
        pipe.vae.decoder,
        name="vae",
    )

    return pipe


def get_dataloader() -> List[Tuple[int, List[str]]]:
    return [
        (1, ["a photo of an astronaut riding a horse on mars"]),
        (1, ["a dog"]),
    ]


def get_config():
    import model_navigator as nav

    return nav.OptimizeConfig(
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


def main():
    import diffusers  # pytype: disable=import-error
    import torch  # pytype: disable=import-error
    import transformers  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.inplace.registry import module_registry
    from tests.functional.common.utils import collect_optimize_statuses, validate_status

    device = torch.device("cuda")
    transformers.modeling_utils.get_parameter_device = lambda parameter: device
    diffusers.models.modeling_utils.get_parameter_device = lambda parameter: device

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    args = parser.parse_args()

    logger.debug(f"CLI args: {args}")

    pipe = get_pipeline()
    dataloader = get_dataloader()
    config = get_config()

    optimize_status = nav.optimize(pipe, dataloader, config=config)
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
    profile_status = nav.profile(pipe, dataloader, window_size=5, verbose=True)
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
