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
"""Find Max Batch size pipelines builders."""

import pathlib
from typing import Dict, List

from model_navigator.commands.base import ExecutionUnit
from model_navigator.commands.find_max_batch_size import FindMaxBatchSize, FindMaxBatchSizeConfig
from model_navigator.configuration import DeviceKind, Format
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.core.logger import LOGGER
from model_navigator.exceptions import ModelNavigatorRuntimeError
from model_navigator.frameworks import Framework
from model_navigator.pipelines.constants import PIPELINE_FIND_MAX_BATCH_SIZE
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.runners.onnx import OnnxrtCPURunner, OnnxrtCUDARunner
from model_navigator.runners.tensorflow import TensorFlowSavedModelCPURunner, TensorFlowSavedModelCUDARunner
from model_navigator.runners.torch import TorchCPURunner, TorchCUDARunner
from model_navigator.utils.config_helpers import do_find_device_max_batch_size


def find_device_max_batch_size_builder(
    config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]
) -> Pipeline:
    """Build finding max batch size for fixed shapes based pipeline.

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        Pipeline with steps for find max batch size.
    """
    execution_units: List[ExecutionUnit] = []

    if not do_find_device_max_batch_size(config=config, models_config=models_config):
        return Pipeline(name=PIPELINE_FIND_MAX_BATCH_SIZE, execution_units=execution_units)

    if config.framework == Framework.TORCH:
        LOGGER.debug("Preparing find max batch size for Torch.")
        configurations = _find_max_batch_size_config_for_torch(config=config, models_config=models_config)
    elif config.framework == Framework.TENSORFLOW:
        LOGGER.debug("Preparing find max batch size for TensorFlow.")
        configurations = _find_max_batch_size_config_for_tensorflow(config=config, models_config=models_config)
    elif config.framework == Framework.JAX:
        LOGGER.debug("Preparing find max batch size for JAX.")
        configurations = _find_max_batch_size_config_for_tensorflow(config=config, models_config=models_config)
    elif config.framework == Framework.ONNX:
        LOGGER.debug("Preparing find max batch size for ONNX.")
        configurations = _find_max_batch_size_config_for_onnx(config=config, models_config=models_config)
    else:
        configurations = []

    execution_units.append(
        ExecutionUnit(
            command=FindMaxBatchSize,
            configurations=configurations,
        )
    )

    return Pipeline(name=PIPELINE_FIND_MAX_BATCH_SIZE, execution_units=execution_units)


def _find_max_batch_size_config_for_torch(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]):
    configurations = []
    for model_cfg in models_config.get(Format.TORCH, []):
        runner_cls = {
            DeviceKind.CUDA: TorchCUDARunner,
            DeviceKind.CPU: TorchCPURunner,
        }[config.target_device]

        if model_cfg.format != runner_cls.format():
            raise ModelNavigatorRuntimeError(
                f"Model config format `{model_cfg.format}` does not match `{runner_cls.format()}`."
            )
        mbs_config = FindMaxBatchSizeConfig(
            format=Format.TORCH,
            model=config.model,
            runner_cls=runner_cls,
            reproduction_scripts_dir=pathlib.Path(model_cfg.key),
        )
        configurations.append(mbs_config)
    for model_cfg in models_config.get(Format.ONNX, []):
        runner_cls = {
            DeviceKind.CUDA: OnnxrtCUDARunner,
            DeviceKind.CPU: OnnxrtCPURunner,
        }[config.target_device]

        if model_cfg.format != runner_cls.format():
            raise ModelNavigatorRuntimeError(
                f"Model config format `{model_cfg.format}` does not match `{runner_cls.format()}`."
            )
        mbs_config = FindMaxBatchSizeConfig(
            format=Format.ONNX,
            model_path=model_cfg.path,
            runner_cls=runner_cls,
            reproduction_scripts_dir=pathlib.Path(model_cfg.key),
        )
        configurations.append(mbs_config)

    return configurations


def _find_max_batch_size_config_for_tensorflow(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]):
    configurations = []
    for model_cfg in models_config.get(Format.TF_SAVEDMODEL, []):
        runner_cls = {
            DeviceKind.CUDA: TensorFlowSavedModelCUDARunner,
            DeviceKind.CPU: TensorFlowSavedModelCPURunner,
        }[config.target_device]

        if model_cfg.format != runner_cls.format():
            raise ModelNavigatorRuntimeError(
                f"Model config format `{model_cfg.format}` does not match `{runner_cls.format()}`."
            )
        mbs_config = FindMaxBatchSizeConfig(
            format=Format.TF_SAVEDMODEL,
            model_path=model_cfg.path,
            runner_cls=runner_cls,
            reproduction_scripts_dir=pathlib.Path(model_cfg.key),
        )
        configurations.append(mbs_config)
    for model_cfg in models_config.get(Format.ONNX, []):
        runner_cls = {
            DeviceKind.CUDA: OnnxrtCUDARunner,
            DeviceKind.CPU: OnnxrtCPURunner,
        }[config.target_device]

        if model_cfg.format != runner_cls.format():
            raise ModelNavigatorRuntimeError(
                f"Model config format `{model_cfg.format}` does not match `{runner_cls.format()}`."
            )
        mbs_config = FindMaxBatchSizeConfig(
            format=Format.ONNX,
            model_path=model_cfg.path,
            runner_cls=runner_cls,
            reproduction_scripts_dir=pathlib.Path(model_cfg.key),
        )
        configurations.append(mbs_config)

    return configurations


def _find_max_batch_size_config_for_onnx(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]):
    configurations = []
    for model_cfg in models_config.get(Format.ONNX, []):
        runner_cls = {
            DeviceKind.CUDA: OnnxrtCUDARunner,
            DeviceKind.CPU: OnnxrtCPURunner,
        }[config.target_device]

        if model_cfg.format != runner_cls.format():
            raise ModelNavigatorRuntimeError(
                f"Model config format `{model_cfg.format}` does not match `{runner_cls.format()}`."
            )
        mbs_config = FindMaxBatchSizeConfig(
            format=Format.ONNX,
            model_path=model_cfg.path,
            runner_cls=runner_cls,
            reproduction_scripts_dir=pathlib.Path(model_cfg.key),
        )
        configurations.append(mbs_config)

    return configurations
