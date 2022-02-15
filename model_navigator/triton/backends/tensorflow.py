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
from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.model import Format
from model_navigator.triton import TritonModelOptimizationConfig
from model_navigator.triton.backends.base import BaseBackendConfigurator
from model_navigator.triton.config import BackendAccelerator


class TensorFlowBackendConfigurator(BaseBackendConfigurator):
    backend_name = "tensorflow"
    supported_formats = [Format.TF_SAVEDMODEL]

    def _set_backend_acceleration(
        self,
        model_config,
        optimization_config: TritonModelOptimizationConfig,
        tensorrt_common_config: TensorRTCommonConfig,
    ):
        if optimization_config.backend_accelerator == BackendAccelerator.AMP:
            accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "auto_mixed_precision"
