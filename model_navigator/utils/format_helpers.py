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

from typing import Optional, Set

from model_navigator.api.config import Format
from model_navigator.utils.framework import Framework


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
    )


def get_framework_export_formats(framework: Framework) -> Set[Format]:
    """Get the base export formats for framework.

    The base export format is the one which can be generated directly from the Python sources.

    Args:
        framework: A framework for which the base format has to be obtained

    Returns:
        Set with supported export formats for given framework.
    """
    return {
        Framework.TORCH: {Format.TORCHSCRIPT, Format.ONNX},
        Framework.TENSORFLOW: {
            Format.TF_SAVEDMODEL,
        },
        Framework.ONNX: {Format.ONNX},
        Framework.JAX: {Format.TF_SAVEDMODEL},
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
        Framework.TORCH: {
            Format.TENSORRT: Format.ONNX,
            Format.TORCH_TRT: Format.TORCHSCRIPT,
        },
        Framework.TENSORFLOW: {
            Format.ONNX: Format.TF_SAVEDMODEL,
            Format.TENSORRT: Format.TF_SAVEDMODEL,
            Format.TF_TRT: Format.TF_SAVEDMODEL,
        },
        Framework.ONNX: {Format.TENSORRT: Format.ONNX},
        Framework.JAX: {
            Format.ONNX: Format.TF_SAVEDMODEL,
            Format.TENSORRT: Format.TF_SAVEDMODEL,
            Format.TF_TRT: Format.TF_SAVEDMODEL,
        },
    }[framework].get(format)


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
    },
}

FRAMEWORK2BASE_FORMAT = {
    Framework.TORCH: Format.TORCH,
    Framework.JAX: Format.JAX,
    Framework.TENSORFLOW: Format.TENSORFLOW,
    Framework.ONNX: Format.ONNX,
}
