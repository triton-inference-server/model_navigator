# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Builders for Torch based models."""
from typing import Dict, List

from model_navigator.api.config import DeviceKind, Format
from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.convert.onnx import ConvertONNX2TRT
from model_navigator.commands.convert.torch import ConvertTorchScript2ONNX, ConvertTorchScript2TorchTensorRT
from model_navigator.commands.export.torch import ExportTorch2ONNX, ExportTorch2TorchScript
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.pipelines.pipeline import Pipeline


def torch_export_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Prepare export steps for pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for export
    """
    execution_units: List[ExecutionUnit] = []
    for model_cfg in models_config.get(Format.TORCHSCRIPT, []):
        execution_units.append(ExecutionUnit(command=ExportTorch2TorchScript, model_config=model_cfg))

    for model_cfg in models_config.get(Format.ONNX, []):
        if model_cfg.parent_path in (None, Format.TORCH):
            execution_units.append(ExecutionUnit(command=ExportTorch2ONNX, model_config=model_cfg))

    return Pipeline(name="PyTorch Export", execution_units=execution_units)


def torch_conversion_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Prepare conversions steps for pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for conversion
    """
    execution_units: List[ExecutionUnit] = []
    for model_cfg in models_config.get(Format.ONNX, []):
        if (
            model_cfg.parent_path and Format.TORCHSCRIPT.value in model_cfg.parent_path.as_posix()
        ):  # FIXME find better way to distinguish ONNX from source from ONNX from TorchScript
            execution_units.append(ExecutionUnit(command=ConvertTorchScript2ONNX, model_config=model_cfg))
    if config.target_device == DeviceKind.CUDA:
        for model_cfg in models_config.get(Format.TORCH_TRT, []):
            execution_units.append(ExecutionUnit(command=ConvertTorchScript2TorchTensorRT, model_config=model_cfg))
        for model_cfg in models_config.get(Format.TENSORRT, []):
            execution_units.append(ExecutionUnit(command=ConvertONNX2TRT, model_config=model_cfg))

    return Pipeline(name="PyTorch Conversion", execution_units=execution_units)
