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
"""Find Max Batch size pipelines builders."""

from typing import Dict, List

from model_navigator.api.config import Format
from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.find_max_batch_size import FindMaxBatchSize, FindMaxBatchSizeConfig
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.core.logger import LOGGER
from model_navigator.frameworks import Framework
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.runners.onnx import OnnxrtCUDARunner
from model_navigator.runners.tensorflow import TensorFlowSavedModelCUDARunner
from model_navigator.runners.torch import TorchScriptCUDARunner
from model_navigator.utils.config_helpers import do_find_device_max_batch_size


def find_device_max_batch_size_builder(
    config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]
) -> Pipeline:
    """Build find device max batch size pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for find max batch size.
    """
    pipeline_name = "Find Device Max Batch Size"
    execution_units: List[ExecutionUnit] = []

    if not do_find_device_max_batch_size(config=config, models_config=models_config):
        return Pipeline(name=pipeline_name, execution_units=execution_units)

    configurations = []
    if config.framework == Framework.TORCH:
        LOGGER.debug("Preparing find max batch size for Torch.")
        for model_cfg in models_config.get(Format.TORCHSCRIPT, []):
            runner_cls = TorchScriptCUDARunner
            assert model_cfg.format == runner_cls.format()
            mbs_config = FindMaxBatchSizeConfig(
                model_path=model_cfg.path,
                runner_cls=runner_cls,
            )
            configurations.append(mbs_config)

        for model_cfg in models_config.get(Format.ONNX, []):
            runner_cls = OnnxrtCUDARunner
            assert model_cfg.format == OnnxrtCUDARunner.format()
            mbs_config = FindMaxBatchSizeConfig(
                model_path=model_cfg.path,
                runner_cls=runner_cls,
            )
            configurations.append(mbs_config)
    elif config.framework == Framework.TENSORFLOW:
        LOGGER.debug("Preparing find max batch size for TensorFlow.")
        for model_cfg in models_config.get(Format.TF_SAVEDMODEL, []):
            runner_cls = TensorFlowSavedModelCUDARunner
            assert model_cfg.format == runner_cls.format()
            mbs_config = FindMaxBatchSizeConfig(
                model_path=model_cfg.path,
                runner_cls=runner_cls,
            )
            configurations.append(mbs_config)
    elif config.framework == Framework.JAX:
        LOGGER.debug("Preparing find max batch size for JAX.")
        for model_cfg in models_config.get(Format.TF_SAVEDMODEL, []):
            runner_cls = TensorFlowSavedModelCUDARunner
            assert model_cfg.format == runner_cls.format()
            mbs_config = FindMaxBatchSizeConfig(
                model_path=model_cfg.path,
                runner_cls=runner_cls,
            )
            configurations.append(mbs_config)
    elif config.framework == Framework.ONNX:
        LOGGER.debug("Preparing find max batch size for ONNX.")
        for model_cfg in models_config.get(Format.ONNX, []):
            runner_cls = OnnxrtCUDARunner
            assert model_cfg.format == runner_cls.format()
            mbs_config = FindMaxBatchSizeConfig(
                model_path=model_cfg.path,
                runner_cls=runner_cls,
            )
            configurations.append(mbs_config)

    execution_units.append(
        ExecutionUnit(
            command=FindMaxBatchSize,
            configurations=configurations,
        )
    )

    return Pipeline(name=pipeline_name, execution_units=execution_units)
