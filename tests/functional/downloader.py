# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import ParseResult

from model_navigator.framework import PyTorch, TensorFlow2

LOGGER = logging.getLogger("model_downloader")

# TODO: refactor this, its duplicated
_SUFFIX2FRAMEWORK = {
    ".savedmodel": TensorFlow2,
    ".plan": PyTorch,
    ".onnx": PyTorch,
    ".pt": PyTorch,
}


class ModelDownloader(ABC):
    def __init__(self, global_config):
        self._global_config = global_config
        model_path = Path(global_config["model_path"])
        self.framework = _SUFFIX2FRAMEWORK[model_path.suffix]
        self.mounts = []
        self.envs = {}

    @abstractmethod
    def download_model(self, url: ParseResult, output_path: Path, **model_kwargs):
        pass
