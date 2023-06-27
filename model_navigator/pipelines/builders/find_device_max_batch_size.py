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
from typing import Dict, List, Union

from model_navigator.api.config import DeviceKind, Format
from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.find_max_batch_size import FindMaxBatchSize, FindMaxBatchSizeConfig
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import (
    ModelConfig,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TorchTensorRTConfig,
)
from model_navigator.core.logger import LOGGER
from model_navigator.frameworks import Framework
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.runners.onnx import OnnxrtCUDARunner
from model_navigator.runners.tensorflow import TensorFlowSavedModelCUDARunner
from model_navigator.runners.torch import TorchScriptCUDARunner


def do_run_max_batch_size_search(
    config: CommonConfig,
    model_cfg: Union[TensorRTConfig, TensorFlowTensorRTConfig, TorchTensorRTConfig],
) -> bool:
    """Should max batch size search be run for the model.

    Args:
        config: Common optimize configuration.
        model_cfg: Model configuration.

    Returns:
        bool: True if run max batch size.
    """
    return bool(model_cfg.trt_profile) is False and config.batch_dim is not None


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
    model_formats = models_config.keys()
    adaptive_formats = {Format.TORCH_TRT, Format.TENSORRT, Format.TF_TRT}
    matching_formats = adaptive_formats.intersection(set(model_formats))

    if len(matching_formats) == 0 or config.target_device != DeviceKind.CUDA:
        LOGGER.debug("No matching formats found")
        return Pipeline(name=pipeline_name, execution_units=execution_units)

    run_search = False
    for fmt in adaptive_formats:
        for model_cfg in models_config.get(fmt, []):
            if do_run_max_batch_size_search(config, model_cfg):
                run_search = True

    if not run_search:
        LOGGER.debug("Run search disabled.")
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
