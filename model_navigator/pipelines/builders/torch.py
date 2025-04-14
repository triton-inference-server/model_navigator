# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

from copy import deepcopy
from typing import Dict, List

from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.convert.torch import ConvertTorchScript2ONNX
from model_navigator.commands.copy.copy_model import CopyModelFromPath
from model_navigator.commands.export.torch import (
    ExportExportedProgram,
    ExportOnnxFromQuantizedTorch,
    ExportTorch2ONNX,
    ExportTorch2TorchScript,
)
from model_navigator.commands.optimize.graph_surgeon import GraphSurgeonOptimize
from model_navigator.configuration import Format, TensorRTPrecision
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig, ONNXModelConfig
from model_navigator.pipelines.constants import (
    PIPELINE_TORCH_CONVERSION,
    PIPELINE_TORCH_EXPORT,
    PIPELINE_TORCH_EXPORTEDPROGRAM,
)
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

    return Pipeline(name=PIPELINE_TORCH_EXPORT, execution_units=execution_units)


def torch_export_onnx_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Prepare export steps for pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for export
    """
    execution_units: List[ExecutionUnit] = []
    for model_cfg in models_config.get(Format.ONNX, []):
        if model_cfg.parent_path in (None, Format.TORCH):
            #  If model_path provided in onnx config, copy this onnx instead of exporting.
            if model_cfg.model_path:  # pytype: disable=attribute-error
                execution_units.append(
                    ExecutionUnit(command=CopyModelFromPath, model_config=models_config[Format.ONNX][0])
                )
            else:
                execution_units.append(ExecutionUnit(command=ExportTorch2ONNX, model_config=model_cfg))

            for model_cfg_trt in models_config.get(Format.TENSORRT, []):
                if model_cfg_trt.precision == TensorRTPrecision.NVFP4:  # pytype: disable=attribute-error
                    # Use trt config copy for special case of NVFP4 torch->model opt quantization->onnx->trt conversion.
                    # Patch it with opset from onnx config.
                    trt_cfg_copy = deepcopy(model_cfg_trt)
                    trt_cfg_copy.opset = model_cfg.opset  # pytype: disable=attribute-error
                    execution_units.append(
                        ExecutionUnit(command=ExportOnnxFromQuantizedTorch, model_config=trt_cfg_copy)
                    )
            assert isinstance(model_cfg, ONNXModelConfig)
            if model_cfg.graph_surgeon_optimization:
                execution_units.append(ExecutionUnit(command=GraphSurgeonOptimize, model_config=model_cfg))

    return Pipeline(name=PIPELINE_TORCH_EXPORT, execution_units=execution_units)


def torch_exportedprogram_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Prepare export steps for pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for export
    """
    execution_units: List[ExecutionUnit] = []

    for model_cfg in models_config.get(Format.TORCH_EXPORTEDPROGRAM, []):
        execution_units.append(ExecutionUnit(command=ExportExportedProgram, model_config=model_cfg))

    return Pipeline(name=PIPELINE_TORCH_EXPORTEDPROGRAM, execution_units=execution_units)


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
            assert isinstance(model_cfg, ONNXModelConfig)
            if model_cfg.graph_surgeon_optimization:
                execution_units.append(ExecutionUnit(command=GraphSurgeonOptimize, model_config=model_cfg))

    return Pipeline(name=PIPELINE_TORCH_CONVERSION, execution_units=execution_units)
