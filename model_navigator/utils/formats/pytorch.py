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
import logging
from dataclasses import dataclass
from pathlib import Path

from model_navigator.model import ModelSignatureConfig
from model_navigator.utils.config import YamlConfigFile
from model_navigator.utils.formats.base import BaseFormatUtils

LOGGER = logging.getLogger(__name__)


@dataclass
class TorchScriptProperties:
    pass


class TorchScriptUtils(BaseFormatUtils):
    @classmethod
    def get_signature(cls, path: Path):
        yaml_path = path.parent / f"{path.name}.yaml"

        try:
            with YamlConfigFile(yaml_path) as config_file:
                signature_config = config_file.load(ModelSignatureConfig)
                cls.validate_signature(signature_config)
                return signature_config
        except OSError:
            raise ValueError(f"TorchScript model configurator expects file {yaml_path} with tensor information.")

    @classmethod
    def validate_signature(cls, signature: ModelSignatureConfig):
        pass

    @classmethod
    def get_properties(cls, path: Path):
        return TorchScriptProperties()
