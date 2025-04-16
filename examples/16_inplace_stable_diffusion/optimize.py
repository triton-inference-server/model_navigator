#!/usr/bin/env python3
# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

import logging

# pytype: disable=import-error
import diffusers
import torch
import transformers
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers.modeling_outputs import BaseModelOutputWithPooling

import model_navigator as nav

# pytype: enable=import-error


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda")

# workaround to make transformers use the same device as model navigator
transformers.modeling_utils.get_parameter_device = lambda parameter: DEVICE
diffusers.models.modeling_utils.get_parameter_device = lambda parameter: DEVICE


def get_pipeline():
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)

    # For outputs that are not primitive types (float, int, bool, str) or tensors and list, dict, tuples combinations of those.
    # we need to provide a mapping to a desired output type. CLIP output is BaseModelOutputWithPooling, which inherits from dict.
    # Model Navigator will recognize that the return type is a dict and will return it, but we need to provide a mapping to BaseModelOutputWithPooling.
    def clip_output_mapping(output):
        return BaseModelOutputWithPooling(**output)

    pipe.text_encoder = nav.Module(
        pipe.text_encoder,
        name="clip",
        output_mapping=clip_output_mapping,
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


def get_dataloader(batched=False):
    return [(1, "a photo of an astronaut riding a horse on mars")]


def get_config():
    return nav.OptimizeConfig(
        batching=False,
        target_formats=(nav.Format.TENSORRT,),
        runners=(
            "TorchCUDA",
            "TensorRT",
            "TorchScriptCUDA",
        ),
        custom_configs=[nav.TensorRTConfig(precision=nav.TensorRTPrecision.FP16), nav.TorchScriptConfig(strict=False)],
    )


def main():
    pipe = get_pipeline()
    dataloader = get_dataloader()
    config = get_config()

    nav.optimize(pipe, dataloader, config=config)

    nav.load_optimized()

    image = pipe(dataloader[0][1]).images[0]
    image.save("astronaut_rides_horse.png")


if __name__ == "__main__":
    main()
