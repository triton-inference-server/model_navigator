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

import logging
import os
import time

# pytype: disable=import-error
import diffusers
import torch
import transformers
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers.modeling_outputs import BaseModelOutputWithPooling

import model_navigator as nav

# pytype: enable=import-error


nav.inplace_config.mode = os.environ.get("MODEL_NAVIGATOR_INPLACE_MODE", nav.inplace_config.mode)
nav.inplace_config.min_num_samples = int(
    os.environ.get("MODEL_NAVIGATOR_MIN_NUM_SAMPLES", nav.inplace_config.min_num_samples)
)


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

    optimize_config = nav.OptimizeConfig(
        batching=False,
        target_formats=(nav.Format.TENSORRT,),
        runners=(
            "TorchCUDA",
            "TorchScriptCUDA",
            "TensorRT",
        ),
        custom_configs=[nav.TensorRTConfig(precision=nav.TensorRTPrecision.FP16), nav.TorchScriptConfig(strict=False)],
    )

    # For outputs that are not primitive types (float, int, bool, str) or tensors and list, dict, tuples combinations of those.
    # we need to provide a mapping to a desired output type. CLIP output is BaseModelOutputWithPooling, which inherits from dict.
    # Model Navigator will recognize that the return type is a dict and will return it, but we need to provide a mapping to BaseModelOutputWithPooling.
    def clip_output_mapping(output):
        return BaseModelOutputWithPooling(**output)

    pipe.text_encoder = nav.Module(
        pipe.text_encoder,
        optimize_config=optimize_config,
        output_mapping=clip_output_mapping,
    )
    pipe.unet = nav.Module(
        pipe.unet,
        optimize_config=optimize_config,
    )
    pipe.vae.decoder = nav.Module(
        pipe.vae.decoder,
        optimize_config=optimize_config,
    )

    return pipe


def get_dataloader():
    return ["a photo of an astronaut riding a horse on mars"]


def main():
    pipe = get_pipeline()
    dataloader = get_dataloader()

    start = time.monotonic()
    image = pipe(dataloader[0]).images[0]
    end = time.monotonic()
    LOGGER.info(f"Elapsed time: {end - start:.2f} seconds")

    image.save(f"astronaut_rides_horse_{nav.inplace_config.mode.value}.png")  # pytype: disable=attribute-error


if __name__ == "__main__":
    main()
