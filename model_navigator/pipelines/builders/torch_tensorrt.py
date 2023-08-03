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
"""Pipeline builders for Torch TensorRT models."""

from typing import Dict, List

from model_navigator.api.config import DeviceKind, Format
from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.convert.torch import ConvertTorchScript2TorchTensorRT
from model_navigator.commands.delete_model import DeleteModel
from model_navigator.commands.performance.performance import Performance
from model_navigator.commands.tensorrt_profile_builder import TensorRTProfileBuilder
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.frameworks.tensorrt.utils import search_for_optimized_profiles
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.runners.registry import get_runner
from model_navigator.runners.torch import TorchTensorRTRunner


def torch_tensorrt_conversion_builder(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> Pipeline:
    """Prepare conversion steps for pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for conversion
    """
    if config.target_device != DeviceKind.CUDA:
        return Pipeline(name="Torch-TensorRT Conversion", execution_units=[])

    torch_trt_models_config = models_config.get(Format.TORCH_TRT, [])
    run_profiles_search = search_for_optimized_profiles(config, torch_trt_models_config)

    execution_units: List[ExecutionUnit] = []
    for model_cfg in torch_trt_models_config:
        if run_profiles_search:
            # Run initial conversion to Torch TensorRT
            execution_units.append(ExecutionUnit(command=ConvertTorchScript2TorchTensorRT, model_config=model_cfg))

            # Generate preliminary profiling results
            execution_units.append(
                ExecutionUnit(
                    command=Performance,
                    model_config=model_cfg,
                    runner_cls=get_runner(TorchTensorRTRunner),
                )
            )

            # Delete temporary Torch TensorRT models
            execution_units.append(ExecutionUnit(command=DeleteModel, model_config=model_cfg))

        # Generate TensorRT profiles or use user provided ones
        execution_units.append(
            ExecutionUnit(
                command=TensorRTProfileBuilder,
                model_config=model_cfg,
                results_lookup_runner_cls=get_runner(TorchTensorRTRunner),
            )
        )

        # Convert TorchScript to Torch TensorRT again, this time with optimized profiles
        execution_units.append(
            ExecutionUnit(
                command=ConvertTorchScript2TorchTensorRT,
                model_config=model_cfg,
                results_lookup_runner_cls=get_runner(TorchTensorRTRunner),
            )
        )
    return Pipeline(name="Torch-TensorRT Conversion", execution_units=execution_units)
