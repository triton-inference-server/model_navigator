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
"""Helpers functions for formats."""

from typing import List, Optional, Set

from model_navigator.api.config import AVAILABLE_TARGET_FORMATS, DEFAULT_TARGET_FORMATS, Format
from model_navigator.frameworks import Framework


def is_source_format(format: Format) -> bool:
    """Validate if provided format is a Python model.

    Args:
        format: A format of model

    Returns:
        True if format is a Python model, False if is a serialized format.
    """
    return format in (
        Format.JAX,
        Format.TENSORFLOW,
        Format.TORCH,
        Format.PYTHON,
    )


def get_target_formats(framework: Framework, is_source_available: bool):
    """Obtain available target formats for framework.

    Args:
        framework: Framework for which target format has to be prepared
        is_source_available: Flag indicating if source is available

    Returns:
        List of target formats
    """
    if is_source_available:
        target_formats = AVAILABLE_TARGET_FORMATS[framework]
    else:
        target_formats = DEFAULT_TARGET_FORMATS[framework]

    return target_formats


def get_framework_export_formats(framework: Framework) -> Set[Optional[Format]]:
    """Get the base export formats for framework.

    The base export format is the one which can be generated directly from the Python sources.
    Python based models cannot be serialized to any format.

    Args:
        framework: A framework for which the base format has to be obtained

    Returns:
        Set with supported export formats for given framework.
    """
    return {
        Framework.NONE: set(),
        Framework.TORCH: {Format.TORCHSCRIPT, Format.ONNX},
        Framework.TENSORFLOW: {
            Format.TF_SAVEDMODEL,
        },
        Framework.ONNX: {Format.ONNX},
        Framework.JAX: {Format.TF_SAVEDMODEL},
        Framework.TENSORRT: {Format.TENSORRT},
    }[framework]


def get_base_format(format: Format, framework: Framework) -> Optional[Format]:
    """Get the base model format required to produce the model in provided format using the framework environment.

    Args:
        format: The format that has to be created
        framework: The framework in which conversion is performed

    Returns:
        A base model format necessary to create provided format.
    """
    return {
        Framework.NONE: {
            Format.PYTHON: Format.PYTHON,
        },
        Framework.TORCH: {
            Format.TENSORRT: Format.ONNX,
            Format.TORCH_TRT: Format.TORCHSCRIPT,
        },
        Framework.TENSORFLOW: {
            Format.ONNX: Format.TF_SAVEDMODEL,
            Format.TENSORRT: Format.ONNX,
            Format.TF_TRT: Format.TF_SAVEDMODEL,
        },
        Framework.ONNX: {Format.TENSORRT: Format.ONNX},
        Framework.JAX: {
            Format.ONNX: Format.TF_SAVEDMODEL,
            Format.TENSORRT: Format.ONNX,
            Format.TF_TRT: Format.TF_SAVEDMODEL,
        },
        Framework.TENSORRT: {Format.TENSORRT: Format.TENSORRT},
    }[framework].get(format)


def get_export_formats(format: Format, framework: Framework) -> List[Format]:
    """Get the export formats required to for max batch size search.

    Args:
        format: The format that has to be created
        framework: The framework in which conversion is performed

    Returns:
        A export model format necessary to create provided format.
    """
    return {
        Framework.NONE: {},
        Framework.TORCH: {
            Format.TENSORRT: [Format.TORCHSCRIPT, Format.ONNX],
            Format.TORCH_TRT: [Format.TORCHSCRIPT],
        },
        Framework.TENSORFLOW: {
            Format.ONNX: [Format.TF_SAVEDMODEL],
            Format.TENSORRT: [Format.TF_SAVEDMODEL],
            Format.TF_TRT: [Format.TF_SAVEDMODEL],
        },
        Framework.ONNX: {},
        Framework.JAX: {
            Format.ONNX: [Format.TF_SAVEDMODEL],
            Format.TENSORRT: [Format.TF_SAVEDMODEL],
            Format.TF_TRT: [Format.TF_SAVEDMODEL],
        },
        Framework.TENSORRT: {},
    }[framework].get(format, [])


SUFFIX2FORMAT = {
    ".savedmodel": Format.TF_SAVEDMODEL,
    ".plan": Format.TENSORRT,
    ".onnx": Format.ONNX,
    ".pt": Format.TORCHSCRIPT,
}

FORMAT2SUFFIX = {format_: suffix for suffix, format_ in SUFFIX2FORMAT.items()}
FORMAT2SUFFIX = {
    **FORMAT2SUFFIX,
    **{
        Format.TF_TRT: ".savedmodel",
        Format.TORCH_TRT: ".pt",
        Format.TORCH_EXPORTEDPROGRAM: ".pt2",
    },
}

FRAMEWORK2BASE_FORMAT = {
    Framework.NONE: Format.PYTHON,
    Framework.TORCH: Format.TORCH,
    Framework.JAX: Format.JAX,
    Framework.TENSORFLOW: Format.TENSORFLOW,
    Framework.ONNX: Format.ONNX,
    Framework.TENSORRT: Format.TENSORRT,
}
