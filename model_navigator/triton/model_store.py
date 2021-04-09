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
from typing import Union

import logging
import shutil
from pathlib import Path

from .. import Format
from .model_config import ModelConfig

LOGGER = logging.getLogger(__name__)

_SUFFIXES = {
    Format.TF_GRAPHDEF: "graphdef",
    Format.TF_SAVEDMODEL: "savedmodel",
    Format.TF_TRT: "savedmodel",
    Format.TRT: "plan",
    Format.ONNX: "onnx",
    Format.TS_TRACE: "pt",
    Format.TS_SCRIPT: "pt",
}


class TritonModelStore:
    def __init__(self, model_store_path: Union[str, Path]):
        self._model_store_path = Path(model_store_path)

    def deploy_model(self, *, model_config: ModelConfig, model_path: Union[str, Path]) -> Path:
        src_model_path = Path(model_path)

        LOGGER.info(
            f"Deploying model {src_model_path} in Triton Model Store {self._model_store_path} "
            f"with config {model_config}"
        )

        # Order of model repository files might be important while using Triton server in polling model_control_mode
        self._copy_model(config=model_config, model_path=src_model_path)

        # remove model filename and model version
        model_dir_in_model_store_path = self._get_model_path(config=model_config).parent.parent
        model_config.save(model_dir=model_dir_in_model_store_path)

        return model_dir_in_model_store_path

    def _copy_model(self, config: ModelConfig, model_path: Path):
        dst_path = self._get_model_path(config)
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        LOGGER.debug(f"Copying {model_path} to {dst_path}")
        if model_path.is_file():
            shutil.copy(model_path, dst_path)
        else:
            shutil.copytree(model_path, dst_path)

    def _get_model_path(self, config: ModelConfig) -> Path:
        suffix = _SUFFIXES[config.model_format]
        return self._model_store_path / config.model_name / config.model_version / f"model.{suffix}"
