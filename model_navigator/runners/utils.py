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
"""Helper function for runners."""

from typing import List, Type

from model_navigator.api.config import Format
from model_navigator.core.logger import LOGGER
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import runner_registry
from model_navigator.utils.devices import is_cuda_available
from model_navigator.utils.format_helpers import is_source_format


def default_runners(device_kind: DeviceKind) -> List:
    """Select default runners defined for the process.

    Returns:
        List of default runners
    """
    _default_runners = set()
    for name, runner in runner_registry.items():
        if device_kind in runner.devices_kind():
            _default_runners.add(name)

    return list(_default_runners)


def get_source_default_runners(format: Format) -> List[Type[NavigatorRunner]]:
    """Get default runner for provided model source format.

    Args:
        format: in which model is implemented

    Returns:
        List of runners objects that support provided format

    Raises:
        ValueError if provided format is not a source format
    """
    if format == Format.PYTHON:
        from model_navigator.runners.python import PythonRunner

        return [PythonRunner]
    if format == Format.TORCH:
        from model_navigator.runners.torch import TorchCPURunner, TorchCUDARunner

        return [TorchCUDARunner, TorchCPURunner] if is_cuda_available() else [TorchCPURunner]
    if format == Format.TENSORFLOW:
        from model_navigator.runners.tensorflow import TensorFlowCPURunner, TensorFlowCUDARunner

        return [TensorFlowCUDARunner, TensorFlowCPURunner] if is_cuda_available() else [TensorFlowCPURunner]
    if format == Format.ONNX:
        from model_navigator.runners.onnx import OnnxrtCPURunner, OnnxrtCUDARunner

        return [OnnxrtCUDARunner, OnnxrtCPURunner] if is_cuda_available() else [OnnxrtCPURunner]
    if format == Format.JAX:
        from model_navigator.runners.jax import JAXRunner

        return [JAXRunner]
    raise ValueError(f"Not source format: {format}")


def get_format_default_runners(format: Format) -> List[Type[NavigatorRunner]]:
    """Get default runner for provided model format.

    Args:
        format: in which model is implemented or serialized

    Returns:
        List of runners objects that support provided format

    Raises:
        ValueError if provided format is not a source format
    """
    if is_source_format(format):
        return get_source_default_runners(format)
    runners = []
    for runner_cls in runner_registry.values():
        if runner_cls.format() == format:
            runners.append(runner_cls)
    if runners:
        LOGGER.info(
            f"Using default runners: `{[runner_cls.name() for runner_cls in runners]}` for format `{format.value}`."
        )
        return runners
    raise ValueError(f"No runner available for format `{format.value}`.")
