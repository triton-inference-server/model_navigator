<!--
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Triton Model Navigator HuggingFace Export

`model_navigator.contrib.huggingface` module allows for an easy and painless export of the [HuggingFace models](https://huggingface.co/models) to the `.nav` package. It can be later use with the Triton Model Navigator [OTIS](../../../../docs/optimize_for_triton.md) to generate the optimal [Triton Inference Server](https://github.com/triton-inference-server) configuration.

The API for `model_navigator.contrib.huggingface.torch` and `model_navigator.contrib.huggingface.tensorflow` are exactly the same.


## Basic export

This is a simple example exporting the [DistilBert model](https://huggingface.co/distilbert-base-uncased).

```python
import model_navigator as nav

nav.contrib.huggingface.torch.export(
    model_name="distilbert-base-uncased",
)

# for TensorFlow2

# nav.contrib.huggingface.tensorflow.export(
#     model_name="distilbert-base-uncased",
# )
```

## Custom HuggingFace Dataset

Basic export will use a dummy input to perform the export and measure performance. You can specify a [Hugginface Dataset](https://huggingface.co/datasets) with a preprocessing function:

```python
import model_navigator as nav
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

nav.contrib.huggingface.torch.export(
    model_name="distilbert-base-uncased",
    dataset_name="imdb",
    dataset_preprocessing_function=lambda examples: tokenizer(examples["text"], truncation=True),
)
```

## Custom Dataloader

You can also directly provide a dataloader that will be used to generate samples for export, correctness tests and performance tests.

```python
import model_navigator as nav

dataloader = get_dataloader()

nav.contrib.huggingface.torch.export(
    model_name="distilbert-base-uncased",
    dataloader=dataloader,
)
```

## Custom Models

Currently we provide the export configurations for the following models:

* [ALBERT](https://huggingface.co/docs/transformers/model_doc/albert),
* [BART](https://huggingface.co/docs/transformers/model_doc/bart),
* [BERT](https://huggingface.co/docs/transformers/model_doc/bert),
* [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert),
* [OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2),
* [GPT Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo),
* [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm),
* [MBart](https://huggingface.co/docs/transformers/model_doc/mbart),
* [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta),

on the following tasks:

* sequence-classification,
* question-answering,
* causal-lm,
* masked-lm.

You can still export other models but you have to provide an export configuration by specifying `onnx_config` parameter. Detailed instructions can be found in the [HuggingFace docs](https://huggingface.co/docs/transformers/main_classes/onnx#onnx-configurations). In short you have to implement a `OnnxConfig` class that specifies inputs, outputs and dynamic dimensions.

Here is an example of exporting the [ELECTRA model](https://huggingface.co/google/electra-small-discriminator).

```python
from typing import Mapping, OrderedDict

import model_navigator as nav
from transformers import AutoConfig
from transformers.onnx import OnnxConfig

class ElectraConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("logits", {0: "batch", 1: "sequence"}),
            ]
        )

nav.contrib.huggingface.torch.export(
    "google/electra-small-discriminator",
    onnx_config=ElectraConfig(AutoConfig.from_pretrained("google/electra-small-discriminator")),
)

```
