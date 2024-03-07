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

import copy
import logging

# pytype: disable=import-error
import torch
import transformers
from datasets import load_dataset
from transformers import pipeline
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

import model_navigator as nav

# pytype: enable=import-error


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 8

# workaround to make transformers use the same device as model navigator
transformers.modeling_utils.get_parameter_device = lambda parameter: DEVICE

target_formats = (nav.Format.TENSORRT, nav.Format.ONNX)
runners = ("TensorRT", "OnnxCUDA")


def get_pipeline():
    optimize_config = nav.OptimizeConfig(
        target_formats=target_formats,
        runners=runners,
        custom_configs=[nav.TensorRTConfig(precision=nav.TensorRTPrecision.FP16)],
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


# Whisper is deleting samples, allways return copy for inference
class CopyList(list):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        return copy.deepcopy(item)

    def __iter__(self):
        for item in super().__iter__():
            yield copy.deepcopy(item)


def main():
    LOGGER.info("Loading pipeline...")
    pipe = get_pipeline()
    LOGGER.info("Pipeline loaded")
    LOGGER.info("Loading dataset...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    dataloader = CopyList()
    for i in range(5):
        dataloader.append((BATCH_SIZE, {"inputs": dataset[i]["audio"]}))

    LOGGER.info(f"Dataset loaded. Dataset length: {len(dataset)}")

    LOGGER.info("Optimizing")
    nav.optimize(pipe, dataloader)

    def copy_wrapper(*args, **kwargs):
        return pipe(*args, **copy.deepcopy(kwargs))

    LOGGER.info("Profiling")
    nav.profile(
        copy_wrapper,
        dataloader,
        target_formats=target_formats,
        runners=runners,
        window_size=3,
        max_trials=3,
        min_trials=1,
        stabilization_windows=3,
        stability_percentage=50,
    )


if __name__ == "__main__":
    main()
