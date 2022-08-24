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
from transformers import FlaxT5Model, T5Tokenizer

import model_navigator as nav

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = FlaxT5Model.from_pretrained("t5-small")

input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="np").input_ids
decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids
dataloader = [{"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}]


def call_wrapper(input_ids, decoder_input_ids, params):
    return model.__call__(input_ids=input_ids, decoder_input_ids=decoder_input_ids, params=params)


desc = nav.jax.export(
    model=call_wrapper,
    model_params=model._params,
    dataloader=dataloader,
    override_workdir=True,
)

desc.save(Path.cwd() / "t5.nav")
