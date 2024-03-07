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

# pytype: disable=import-error
import torch
from transformers import pipeline
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

import model_navigator as nav

# pytype: enable=import-error

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/bart-large-cnn"

batch_size = 1
ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
dataloader = [(batch_size, ARTICLE)]

target_formats = (nav.Format.TENSORRT,)
runners = ("TorchCUDA", "TensorRT")

optimize_config = nav.OptimizeConfig(
    batching=False,
    target_formats=target_formats,
    runners=runners,
    custom_configs=[nav.TensorRTConfig(precision=nav.TensorRTPrecision.FP16)],
)


def get_pipeline():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=DEVICE)
    summarizer.model.model.encoder = nav.Module(
        summarizer.model.model.encoder,
        name="encoder",
        output_mapping=lambda x: BaseModelOutput(**x),
    )
    summarizer.model.model.decoder = nav.Module(
        summarizer.model.model.decoder,
        name="decoder",
        output_mapping=lambda x: BaseModelOutputWithPastAndCrossAttentions(**x),
    )
    summarizer.model.lm_head = nav.Module(
        summarizer.model.lm_head,
        name="lm_head",
    )
    return summarizer


def main():
    LOGGER.info("Loading model...")
    pipe = get_pipeline()
    LOGGER.info("Model loaded")

    LOGGER.info("Optimizing")
    nav.optimize(pipe, dataloader, config=optimize_config)

    LOGGER.info("Profiling")
    nav.profile(pipe, dataloader, target_formats=target_formats, runners=runners, window_size=5)


if __name__ == "__main__":
    main()
