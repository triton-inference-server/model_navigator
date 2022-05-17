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
from model_navigator.framework_api.commands.convert.pyt import ConvertTorchScript2TorchTensorRT
from model_navigator.framework_api.commands.core import Command
from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX, ExportPYT2TorchScript
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import Framework
from model_navigator.model import Format

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


def torch_export_builder(config: Config, package_descriptor: "PackageDescriptor") -> Pipeline:
    commands: List[Command] = []
    ts_exports = {}

    if Format.TORCHSCRIPT in config.target_formats or Format.TORCH_TRT in config.target_formats:
        for target_jit_type in config.target_jit_type:
            export = ExportPYT2TorchScript(target_jit_type=target_jit_type)
            commands.append(export)
            ts_exports[target_jit_type] = export
    if Format.ONNX in config.target_formats or Format.TENSORRT in config.target_formats:
        onnx_export = ExportPYT2ONNX()
        commands.append(onnx_export)
    if Format.TORCH_TRT in config.target_formats:
        for target_jit_type in config.target_jit_type:
            commands.append(
                ConvertTorchScript2TorchTensorRT(
                    target_jit_type=target_jit_type, requires=(ts_exports[target_jit_type],)
                )
            )
    if Format.TENSORRT in config.target_formats:
        for target_precision in config.target_precisions:
            commands.append(ConvertONNX2TRT(target_precision=target_precision, requires=(onnx_export,)))

    return Pipeline(name="PyTorch Export", framework=Framework.PYT, commands=commands)
