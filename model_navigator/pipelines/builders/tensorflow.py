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
"""Tensorflow pipelines builders."""

from typing import Dict, List

from model_navigator.api.config import Format
from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.convert.tf import ConvertSavedModel2ONNX
from model_navigator.commands.export.tf import ExportTF2SavedModel
from model_navigator.commands.optimize.graph_surgeon import GraphSurgeonOptimize
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig, ONNXConfig
from model_navigator.pipelines.pipeline import Pipeline


def tensorflow_export_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Build Tensorflow export pipeline.

    Args:
        config (CommonConfig): Common optimize config.
        models_config (Dict[Format, List[ModelConfig]]): Models optimize configs.

    Returns:
        Pipeline: Export pipeline.
    """
    execution_units: List[ExecutionUnit] = []
    for model_cfg in models_config.get(Format.TF_SAVEDMODEL, []):
        execution_units.append(ExecutionUnit(command=ExportTF2SavedModel, model_config=model_cfg))
    return Pipeline(name="TensorFlow2 Export", execution_units=execution_units)


def tensorflow_conversion_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Build Tensorflow conversion pipeline.

    Args:
        config (CommonConfig): Common optimize config.
        models_config (Dict[Format, List[ModelConfig]]): Models optimize configs.

    Returns:
        Pipeline: Conversion pipeline.
    """
    execution_units: List[ExecutionUnit] = []
    for model_cfg in models_config.get(Format.ONNX, []):
        execution_units.append(ExecutionUnit(command=ConvertSavedModel2ONNX, model_config=model_cfg))
        assert isinstance(model_cfg, ONNXConfig)
        if model_cfg.graph_surgeon_optimization:
            execution_units.append(ExecutionUnit(command=GraphSurgeonOptimize, model_config=model_cfg))

    return Pipeline(name="TensorFlow 2 Conversion", execution_units=execution_units)
