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

# pytype: disable=import-error
from transformers.models.albert import AlbertConfig, AlbertOnnxConfig
from transformers.models.bart import BartConfig, BartOnnxConfig
from transformers.models.bert import BertConfig, BertOnnxConfig
from transformers.models.distilbert import DistilBertConfig, DistilBertOnnxConfig
from transformers.models.gpt2 import GPT2Config, GPT2OnnxConfig
from transformers.models.gpt_neo import GPTNeoConfig, GPTNeoOnnxConfig
from transformers.models.layoutlm import LayoutLMConfig, LayoutLMOnnxConfig
from transformers.models.longformer import LongformerConfig, LongformerOnnxConfig
from transformers.models.mbart import MBartConfig, MBartOnnxConfig
from transformers.models.roberta import RobertaConfig, RobertaOnnxConfig
from transformers.models.t5 import T5Config, T5OnnxConfig
from transformers.models.xlm_roberta import XLMRobertaConfig, XLMRobertaOnnxConfig

# pytype: enable=import-error

_MODEL_TO_ONNX_CONFIG = {
    BertConfig: BertOnnxConfig,
    DistilBertConfig: DistilBertOnnxConfig,
    AlbertConfig: AlbertOnnxConfig,
    BartConfig: BartOnnxConfig,
    GPTNeoConfig: GPTNeoOnnxConfig,
    LayoutLMConfig: LayoutLMOnnxConfig,
    LongformerConfig: LongformerOnnxConfig,
    MBartConfig: MBartOnnxConfig,
    GPT2Config: GPT2OnnxConfig,
    RobertaConfig: RobertaOnnxConfig,
    T5Config: T5OnnxConfig,
    XLMRobertaConfig: XLMRobertaOnnxConfig,
}


def get_onnx_config(config):

    for config_class, onnx_config_class in _MODEL_TO_ONNX_CONFIG.items():
        if type(config) == config_class:
            return onnx_config_class(config)

    raise ValueError(f"No default ONNX config for {type(config)}. Plase provide the config.")
