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
"""Preprocessing pipelines builders."""

from typing import Dict, List

from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.copy.copy_model import CopyModel
from model_navigator.commands.data_dump.samples import FetchInputModelData, FetchOutputModelData
from model_navigator.commands.infer_metadata import InferInputMetadata, InferOutputMetadata
from model_navigator.commands.load import LoadMetadata
from model_navigator.configuration import Format
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.pipelines.constants import PIPELINE_PREPROCESSING
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.runners.utils import get_format_default_runners
from model_navigator.utils.format_helpers import FRAMEWORK2BASE_FORMAT


def preprocessing_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Build profiling pipeline.

    When package is loaded using nav.package.load
    config.from_source is False and metadata and samples
    are loaded directly from the drive.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for profiling.
    """
    format = FRAMEWORK2BASE_FORMAT[config.framework]
    model_config = models_config[format][0] if models_config[format] else None
    runners = get_format_default_runners(format)

    execution_units: List[ExecutionUnit] = []
    for runner in runners:
        if config.target_device in runner.devices_kind():
            if config.from_source:
                runner_config = getattr(model_config, "runner_config", None)
                execution_units.extend([
                    ExecutionUnit(command=InferInputMetadata),
                    ExecutionUnit(command=FetchInputModelData),
                    ExecutionUnit(command=InferOutputMetadata, runner_config=runner_config, runner_cls=runner),
                    ExecutionUnit(command=FetchOutputModelData, runner_config=runner_config, runner_cls=runner),
                ])
            else:
                execution_units.extend([ExecutionUnit(command=LoadMetadata)])

            if format in (Format.ONNX, Format.TENSORRT):
                execution_units.append(ExecutionUnit(command=CopyModel, model_config=model_config))

            break

    return Pipeline(name=PIPELINE_PREPROCESSING, execution_units=execution_units)
