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
from model_navigator.utils.formats.base import BaseFormatUtils

LOGGER = logging.getLogger(__name__)


@dataclass
class TensorRTProperties:
    pass


class TensorRTUtils(BaseFormatUtils):
    @classmethod
    def get_signature(cls, path: Path):
        return None

    @classmethod
    def validate_signature(cls, signature: ModelSignatureConfig):
        pass

    @classmethod
    def get_properties(cls, path: Path):
        return TensorRTProperties()
