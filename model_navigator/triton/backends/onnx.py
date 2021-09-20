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
from distutils.version import LooseVersion

from model_navigator.exceptions import BadParameterModelNavigatorDeployerException
from model_navigator.model import Format, Model
from model_navigator.triton import TritonModelOptimizationConfig
from model_navigator.triton.backends.base import BaseBackendConfigurator
from model_navigator.triton.config import BackendAccelerator
from model_navigator.triton.utils import rewrite_signature_to_model_config


class OnnxBackendConfigurator(BaseBackendConfigurator):
    backend_name = "onnx"
    supported_formats = [Format.ONNX]

    def _extract_signature(self, model_config, model: Model):
        # https://github.com/triton-inference-server/onnxruntime_backend/pull/16
        TRITON_VERSION_WITH_FIXED_ONNX_SIGNATURE_EXTRACT = LooseVersion("2.7.0")
        version_of_triton_with_bug = (
            self._target_triton_version is not None
            and LooseVersion(self._target_triton_version) < TRITON_VERSION_WITH_FIXED_ONNX_SIGNATURE_EXTRACT
        )
        if version_of_triton_with_bug:
            rewrite_signature_to_model_config(model_config, model.signature)

    def _set_backend_acceleration(self, model_config, optimization_config: TritonModelOptimizationConfig):
        if optimization_config.backend_accelerator == BackendAccelerator.TRT:
            if optimization_config.tensorrt_precision is None:
                raise BadParameterModelNavigatorDeployerException(
                    "--tensorrt-precision is required while using tensorrt backend accelerator"
                )

            accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "tensorrt"
            accelerator.parameters["precision_mode"] = optimization_config.tensorrt_precision.value.upper()
            accelerator.parameters["max_workspace_size_bytes"] = str(optimization_config.tensorrt_max_workspace_size)
