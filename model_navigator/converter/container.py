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
from model_navigator.framework import PyTorch, TensorFlow2
from model_navigator.model import Format

# Map model formats to frameworks
FORMAT2IMAGE = {
    Format.TF_SAVEDMODEL: TensorFlow2,
    Format.ONNX: PyTorch,
    Format.TENSORRT: PyTorch,
    Format.TORCHSCRIPT: PyTorch,
}


class ConverterContainer:
    @staticmethod
    def base_image(model_format: Format, container_version: str) -> str:
        """
        Create base container image name for converter based on model format and container version

        Args:
            model_format: Model Format
            container_version: xx.xx format container version

        Returns:
            Base container image name
        """
        framework = FORMAT2IMAGE[model_format]
        base_framework_image_name = f"{framework.image}:{container_version}-{framework.tag}"
        return base_framework_image_name

    @staticmethod
    def image(model_format: Format, container_version: str) -> str:
        """
        Create image name for converter based on model format and container version

        Args:
            model_format: Model Format
            container_version: xx.xx format container version

        Returns:
            Container image name
        """
        framework = FORMAT2IMAGE[model_format]
        converter_image_name = f"model_navigator_converter:{container_version}-{framework.tag}"
        return converter_image_name
