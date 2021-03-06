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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from model_navigator.model import ModelSignatureConfig


class BaseFormatUtils(ABC):
    @classmethod
    @abstractmethod
    def get_signature(cls, path: Path) -> ModelSignatureConfig:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def validate_signature(cls, signature: ModelSignatureConfig):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_properties(cls, path: Path) -> Any:
        raise NotImplementedError()

    @classmethod
    def get_num_required_gpus(cls, properties: Any) -> Optional[int]:
        return None
