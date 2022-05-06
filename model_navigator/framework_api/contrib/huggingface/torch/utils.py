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

import importlib
from typing import Optional, Sequence

# pytype: disable=import-error
from transformers import AutoModel, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from model_navigator.framework_api.logger import LOGGER

# pytype: enable=import-error

_CONFIG_MODULE_MAPPING = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def get_max_sequence_length(tokenizer: PreTrainedTokenizer) -> Optional[int]:
    return getattr(tokenizer, "model_max_length", None)


def get_module_from_config(config: PretrainedConfig) -> str:
    module_name = _CONFIG_MODULE_MAPPING.get(config.__class__.__name__, None)
    if module_name is None:
        raise ValueError(f"No module for config of type {type(config)}")
    return module_name


def get_pretrained_model_from_config(
    model_name: str, config: PretrainedConfig, torchscript: bool = False
) -> PreTrainedModel:
    arch = getattr(config, "architectures", None)
    if arch is None or not isinstance(arch, Sequence):
        LOGGER.warning("Architecture not available in config. Using AutoModel.")
        return AutoModel.from_pretrained(model_name, torchscript=True)
    arch = arch[0]
    module_name = get_module_from_config(config)
    model_cls = getattr(importlib.import_module(f"transformers.models.{module_name.replace('-', '_')}"), arch)
    return model_cls.from_pretrained(model_name, torchscript=torchscript)
