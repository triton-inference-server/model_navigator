# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""Generate Triton Model Config object from specialized model configs.

The class provide builder interfaces to create generic ModelConfig from specialized configs

    Typical usage example:

        model_config = ModelConfigBuilder.from_onnx_config(ONNXModelConfig())
"""

import dataclasses
from typing import Dict, Union

from .model_config import ModelConfig
from .specialized_configs import (
    ONNXModelConfig,
    PythonModelConfig,
    PyTorchModelConfig,
    TensorFlowModelConfig,
    TensorRTModelConfig,
)


class ModelConfigBuilder:
    """Generate ModelConfig object from specialized model configs."""

    @classmethod
    def from_onnx_config(cls, model_name: str, model_version: int, onnx_config: ONNXModelConfig):
        """Create generic ModelConfig from specialized ONNX config.

        Args:
            model_name: Name under which model is deployed
            model_version: Version of model that is deployed
            onnx_config: Configuration of selected model type

        Returns:
            Generic ModelConfig object
        """
        data = cls._get_common_data(onnx_config)
        model_config = ModelConfig(
            model_name=model_name,
            model_version=model_version,
            backend=onnx_config.backend,
            **data,
        )
        return model_config

    @classmethod
    def from_pytorch_config(cls, model_name: str, model_version: int, pytorch_config: PyTorchModelConfig):
        """Create generic ModelConfig from specialized Pytorch  config.

        Args:
            model_name: Name under which model is deployed
            model_version: Version of model that is deployed
            pytorch_config: Configuration of selected model type

        Returns:
            Generic ModelConfig object
        """
        data = cls._get_common_data(pytorch_config)
        model_config = ModelConfig(
            model_name=model_name,
            model_version=model_version,
            backend=pytorch_config.backend,
            **data,
        )
        return model_config

    @classmethod
    def from_python_config(cls, model_name: str, model_version: int, python_config: PythonModelConfig):
        """Create generic ModelConfig from specialized Python config.

        Args:
            model_name: Name under which model is deployed
            model_version: Version of model that is deployed
            python_config: Configuration of selected model type

        Returns:
            Generic ModelConfig object
        """
        data = cls._get_common_data(python_config)
        model_config = ModelConfig(
            model_name=model_name,
            model_version=model_version,
            backend=python_config.backend,
            **data,
        )
        return model_config

    @classmethod
    def from_tensorflow_config(cls, model_name: str, model_version: int, tensorflow_config: TensorFlowModelConfig):
        """Create generic ModelConfig from specialized TensorFlow config.

        Args:
            model_name: Name under which model is deployed
            model_version: Version of model that is deployed
            tensorflow_config: Configuration of selected model type

        Returns:
            Generic ModelConfig object
        """
        data = cls._get_common_data(tensorflow_config)
        model_config = ModelConfig(
            model_name=model_name,
            model_version=model_version,
            backend=tensorflow_config.backend,
            **data,
        )
        return model_config

    @classmethod
    def from_tensorrt_config(cls, model_name: str, model_version: int, tensorrt_config: TensorRTModelConfig):
        """Create generic ModelConfig from specialized TensorRT config.

        Args:
            model_name: Name under which model is deployed
            model_version: Version of model that is deployed
            tensorrt_config: Configuration of selected model type

        Returns:
            Generic ModelConfig object
        """
        data = cls._get_common_data(tensorrt_config)
        model_config = ModelConfig(
            model_name=model_name,
            model_version=model_version,
            backend=tensorrt_config.backend,
            **data,
        )
        return model_config

    @classmethod
    def _get_common_data(
        cls,
        config: Union[
            ONNXModelConfig, PythonModelConfig, PyTorchModelConfig, TensorFlowModelConfig, TensorRTModelConfig
        ],
    ) -> Dict:
        data = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
        return data
