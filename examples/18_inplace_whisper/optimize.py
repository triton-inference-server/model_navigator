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

import logging
import os
import time

# pytype: disable=import-error
import torch
import transformers
from datasets import load_dataset
from transformers import pipeline
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

import model_navigator as nav

# pytype: enable=import-error


nav.inplace_config.mode = os.environ.get("MODEL_NAVIGATOR_INPLACE_MODE", nav.inplace_config.mode)
nav.inplace_config.min_num_samples = int(
    os.environ.get("MODEL_NAVIGATOR_MIN_NUM_SAMPLES", nav.inplace_config.min_num_samples)
)


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 8

# workaround to make transformers use the same device as model navigator
transformers.modeling_utils.get_parameter_device = lambda parameter: DEVICE


def get_pipeline():
    optimize_config = nav.OptimizeConfig(
        target_formats=(nav.Format.TENSORRT,),
        runners=("TensorRT",),
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


def main():
    LOGGER.info("Loading pipeline...")
    pipe = get_pipeline()
    LOGGER.info("Pipeline loaded")
    LOGGER.info("Loading dataset...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = [dataset[i] for i in range(5)]
    LOGGER.info(f"Dataset loaded. Dataset length: {len(dataset)}")

    if nav.inplace_config.mode == nav.Mode.RECORDING:
        LOGGER.info("Recording")
        for batch in dataset:
            pipe(batch["audio"].copy(), batch_size=BATCH_SIZE)

        # additional inference to record max output sequence length
        pipe(dataset[0]["audio"].copy(), batch_size=BATCH_SIZE, generate_kwargs={"min_new_tokens": 512})
        LOGGER.info("Optimizing")
        nav.optimize()

    LOGGER.info("Warmup")
    for i, batch in enumerate(dataset):
        pipe(batch["audio"].copy(), batch_size=BATCH_SIZE)
        if (i := i + 1) > 5:
            break

    LOGGER.info("Inference")

    times = []
    for _ in range(10):
        for batch in dataset:
            start = time.monotonic()
            prediction = pipe(batch["audio"].copy(), batch_size=BATCH_SIZE)["text"]
            end = time.monotonic()
            times.append(end - start)
    LOGGER.info(prediction)
    LOGGER.info(f"Inference time: {sum(times):.2f} seconds")


if __name__ == "__main__":
    main()
