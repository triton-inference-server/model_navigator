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
import shutil
from pathlib import Path
from typing import Optional, Union

from model_navigator.model import Format, Model
from model_navigator.triton.config import (
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
    TritonModelSchedulerConfig,
)
from model_navigator.triton.model_config import TritonModelConfigGenerator

LOGGER = logging.getLogger(__name__)

_SUFFIXES = {
    Format.TF_SAVEDMODEL: "savedmodel",
    Format.TENSORRT: "plan",
    Format.ONNX: "onnx",
    Format.TORCHSCRIPT: "pt",
}


class TritonModelStore:
    def __init__(
        self,
        model_store_path: Union[str, Path],
        target_triton_version: Optional[str] = None,
    ):
        self._model_store_path = Path(model_store_path)
        self._target_triton_version = target_triton_version

    def deploy_model(
        self,
        *,
        model: Model,
        model_version: str,
        optimization_config: TritonModelOptimizationConfig,
        scheduler_config: TritonModelSchedulerConfig,
        instances_config: TritonModelInstancesConfig,
    ) -> Path:

        triton_model_config_generator = TritonModelConfigGenerator(
            model=model,
            optimization_config=optimization_config,
            scheduler_config=scheduler_config,
            instances_config=instances_config,
            target_triton_version=self._target_triton_version,
        )
        LOGGER.debug(
            f"Deploying model {model.path} in Triton Model Store {self._model_store_path} "
            f"with optimization config: {optimization_config} and scheduler_config: {scheduler_config}"
        )

        # Order of model repository files might be important while using Triton server in polling model_control_mode
        model_path = self._copy_model(model=model, version=model_version)

        # remove model filename and model version
        model_dir_in_model_store_path = model_path.parent.parent
        triton_model_config_generator.save(model_dir=model_dir_in_model_store_path)

        return model_dir_in_model_store_path

    def _copy_model(self, model: Model, version: str) -> Path:
        dst_path = self._get_model_path(model, version)
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        LOGGER.debug(f"Copying {model.path} to {dst_path}")
        if model.path.is_file():
            shutil.copy(model.path, dst_path)
        else:
            shutil.copytree(model.path, dst_path)
        return dst_path

    def _get_model_path(self, model: Model, version: str) -> Path:
        suffix = _SUFFIXES[model.format]
        return self._model_store_path / model.name / version / f"model.{suffix}"
