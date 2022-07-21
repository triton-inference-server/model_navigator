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

from typing import TYPE_CHECKING, List

from model_navigator.framework_api.commands.convert.onnx import ConvertONNX2TRT
from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2ONNX, ConvertSavedModel2TFTRT
from model_navigator.framework_api.commands.core import Command
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import Framework
from model_navigator.model import Format

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


def tensorflow_export_builder(config: Config, package_descriptor: "PackageDescriptor") -> Pipeline:
    return Pipeline(name="TensorFlow2 Export", framework=Framework.TF2, commands=[ExportTF2SavedModel()])


def tensorflow_conversion_builder(config: Config, package_descriptor: "PackageDescriptor") -> Pipeline:
    commands: List[Command] = []

    if Format.ONNX in config.target_formats or Format.TENSORRT in config.target_formats:
        commands.append(ConvertSavedModel2ONNX())
    if Format.TENSORRT in config.target_formats:
        for target_precision in config.target_precisions:
            commands.append(ConvertONNX2TRT(target_precision=target_precision))
    if Format.TF_TRT in config.target_formats:
        for target_precision in config.target_precisions:
            commands.append(ConvertSavedModel2TFTRT(target_precision=target_precision))

    return Pipeline(name="TensorFlow2 Conversion", framework=Framework.TF2, commands=commands)
