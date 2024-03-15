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
"""Profiling pipelines builders."""

from typing import Dict, List

from model_navigator.api.config import Format
from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.performance import Performance
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.runners.registry import runner_registry
from model_navigator.utils.format_helpers import is_source_format


def performance_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Build performance pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for profiling.
    """
    execution_units: List[ExecutionUnit] = []
    source_last_formats = [
        *[model_format for model_format in models_config.keys() if not is_source_format(model_format)],
        *[model_format for model_format in models_config.keys() if is_source_format(model_format)],
    ]
    for model_format in source_last_formats:
        for model_config in models_config[model_format]:
            for runner in runner_registry.values():
                if (
                    runner.format() == model_config.format
                    and runner.name() in config.runner_names
                    and config.target_device in runner.devices_kind()
                ):
                    execution_units.append(
                        ExecutionUnit(
                            command=Performance,
                            model_config=model_config,
                            runner_cls=runner,
                        )
                    )
    return Pipeline(name="Performance", execution_units=execution_units)
