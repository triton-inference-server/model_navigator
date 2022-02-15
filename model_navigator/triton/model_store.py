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
import shutil
from pathlib import Path
from typing import Optional, Union

from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.exceptions import ModelNavigatorDeployerException
from model_navigator.model import Model
from model_navigator.triton.backends.base import BackendConfiguratorSelector
from model_navigator.triton.config import (
    TritonBatchingConfig,
    TritonCustomBackendParametersConfig,
    TritonDynamicBatchingConfig,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
)
from model_navigator.triton.model_config import TritonModelConfigGenerator

LOGGER = logging.getLogger(__name__)


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
        batching_config: TritonBatchingConfig,
        optimization_config: TritonModelOptimizationConfig,
        tensorrt_common_config: TensorRTCommonConfig,
        dynamic_batching_config: TritonDynamicBatchingConfig,
        instances_config: TritonModelInstancesConfig,
        backend_parameters_config: TritonCustomBackendParametersConfig,
    ) -> Path:

        triton_model_config_generator = TritonModelConfigGenerator(
            model=model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            tensorrt_common_config=tensorrt_common_config,
            dynamic_batching_config=dynamic_batching_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
            target_triton_version=self._target_triton_version,
        )
        LOGGER.debug(
            f"Deploying model {model.path} in Triton Model Store {self._model_store_path} with: "
            f"\nbatching config: {batching_config}"
            f"\noptimization config: {optimization_config}"
            f"\ndynamic batching config: {dynamic_batching_config}"
        )

        # Order of model repository files might be important while using Triton server in polling model_control_mode
        model_path = self._copy_model(model=model, version=model_version)

        # remove model filename and model version
        model_dir_in_model_store_path = model_path.parent.parent
        config_path = model_dir_in_model_store_path / "config.pbtxt"
        triton_model_config_generator.save(config_path=config_path)

        return model_dir_in_model_store_path

    def _copy_model(self, model: Model, version: str) -> Path:
        dst_path = self._get_model_path(model, version)
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        LOGGER.debug(f"Copying {model.path} to {dst_path}")
        if model.path.is_file():
            shutil.copy(model.path, dst_path)
        else:
            try:
                shutil.copytree(model.path, dst_path)
            except shutil.Error:
                # due to error as reported on https://bugs.python.org/issue43743
                shutil._USE_CP_SENDFILE = False
                shutil.rmtree(dst_path)
                shutil.copytree(model.path, dst_path)
        return dst_path

    def _get_model_path(self, model: Model, version: str) -> Path:
        backend_configurator = BackendConfiguratorSelector.for_model(model)
        return self._model_store_path / model.name / version / backend_configurator.get_filename(model)

    def get_model_path(self, model_name):
        model_dir = self._model_store_path / model_name
        # support only single version and single file/directory models
        version_dir_paths = [file_path for file_path in model_dir.iterdir() if file_path.is_dir()]
        if len(version_dir_paths) != 1:
            raise ModelNavigatorDeployerException(
                f"In Triton model directory there is more than 1 model version {version_dir_paths}"
            )
        version_dir_path = version_dir_paths[0]
        model_paths = list(version_dir_path.iterdir())
        if len(model_paths) != 1:
            raise ModelNavigatorDeployerException(f"In Triton model directory there is more than 1 model {model_paths}")
        return model_paths[0].resolve()
