#!/usr/bin/env python3
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path

import tensorflow
from transformers import DistilBertTokenizer, FlaxDistilBertModel

import model_navigator as nav

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = FlaxDistilBertModel.from_pretrained("distilbert-base-uncased")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="np")
dataloader = [encoded_input]


desc = nav.jax.export(
    model=model.__call__,
    model_params=model._params,
    dataloader=dataloader,
    override_workdir=True,
    target_formats=(nav.Format.TF_SAVEDMODEL,),
)

desc.save(Path.cwd() / "distilbert.nav")
