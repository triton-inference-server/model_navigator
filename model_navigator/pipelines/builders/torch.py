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
from model_navigator.commands.find_max_batch_size.find_max_batch_size import FindMaxBatchSize
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.runners.onnx import OnnxrtCUDARunner
from model_navigator.runners.torch import TorchScriptCUDARunner
from model_navigator.utils.pipelines import do_run_max_batch_size_search


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
        execution_units.append(ExecutionUnit(ExportTorch2TorchScript, config, model_cfg))
    for model_cfg in models_config.get(Format.ONNX, []):
        if model_cfg.parent_path in (None, Format.TORCH):
            execution_units.append(ExecutionUnit(ExportTorch2ONNX, config, model_cfg))
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
            execution_units.append(ExecutionUnit(ConvertTorchScript2ONNX, config, model_cfg))
    if config.target_device == DeviceKind.CUDA:
        for model_cfg in models_config.get(Format.TORCH_TRT, []):
            if do_run_max_batch_size_search(config, model_cfg):
                runner_cls = TorchScriptCUDARunner
                assert model_cfg.parent.format == runner_cls.format()
                execution_units.append(
                    ExecutionUnit(
                        FindMaxBatchSize,
                        config,
                        model_cfg.parent,
                        runner_cls=runner_cls,
                    )
                )
            execution_units.append(ExecutionUnit(ConvertTorchScript2TorchTensorRT, config, model_cfg))
        for model_cfg in models_config.get(Format.TENSORRT, []):
            if do_run_max_batch_size_search(config, model_cfg):
                runner_cls = OnnxrtCUDARunner
                assert model_cfg.parent.format == runner_cls.format()
                execution_units.append(
                    ExecutionUnit(
                        FindMaxBatchSize,
                        config,
                        model_cfg.parent,
                        runner_cls=runner_cls,
                    )
                )
            execution_units.append(ExecutionUnit(ConvertONNX2TRT, config, model_cfg))

    return Pipeline(name="PyTorch Conversion", execution_units=execution_units)
