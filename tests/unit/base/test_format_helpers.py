#!/usr/bin/env python3
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

from model_navigator.configuration import Format, Framework
from model_navigator.utils.format_helpers import (
    FORMAT2SUFFIX,
    FRAMEWORK2BASE_FORMAT,
    SUFFIX2FORMAT,
    get_base_format,
    get_framework_export_formats,
    is_source_format,
)


def test_is_source_format_returns_true_when_source_format():
    assert is_source_format(Format.PYTHON)
    assert is_source_format(Format.JAX)
    assert is_source_format(Format.TENSORFLOW)
    assert is_source_format(Format.TORCH)

    assert not is_source_format(Format.TF_SAVEDMODEL)
    assert not is_source_format(Format.TF_TRT)
    assert not is_source_format(Format.TORCHSCRIPT)
    assert not is_source_format(Format.TORCH_TRT)
    assert not is_source_format(Format.ONNX)
    assert not is_source_format(Format.TENSORRT)


def test_get_framework_export_formats_returns_correct_formats():
    assert get_framework_export_formats(Framework.TORCH) == {Format.TORCHSCRIPT, Format.ONNX}
    assert get_framework_export_formats(Framework.TENSORFLOW) == {Format.TF_SAVEDMODEL}
    assert get_framework_export_formats(Framework.ONNX) == {Format.ONNX}
    assert get_framework_export_formats(Framework.JAX) == {Format.TF_SAVEDMODEL}
    assert get_framework_export_formats(Framework.NONE) == set()


def test_get_base_format_returns_correct_formats():
    assert get_base_format(framework=Framework.NONE, format=Format.PYTHON) == Format.PYTHON

    assert get_base_format(framework=Framework.TORCH, format=Format.TENSORRT) == Format.ONNX
    assert get_base_format(framework=Framework.TORCH, format=Format.TORCH_TRT) == Format.TORCH_EXPORTEDPROGRAM

    assert get_base_format(framework=Framework.TENSORFLOW, format=Format.ONNX) == Format.TF_SAVEDMODEL
    assert get_base_format(framework=Framework.TENSORFLOW, format=Format.TENSORRT) == Format.ONNX
    assert get_base_format(framework=Framework.TENSORFLOW, format=Format.TF_TRT) == Format.TF_SAVEDMODEL

    assert get_base_format(framework=Framework.ONNX, format=Format.TENSORRT) == Format.ONNX

    assert get_base_format(framework=Framework.JAX, format=Format.ONNX) == Format.TF_SAVEDMODEL
    assert get_base_format(framework=Framework.JAX, format=Format.TENSORRT) == Format.ONNX
    assert get_base_format(framework=Framework.JAX, format=Format.TF_TRT) == Format.TF_SAVEDMODEL


def test_suffix2format_returns_correct_formats():
    assert SUFFIX2FORMAT[".savedmodel"] == Format.TF_SAVEDMODEL
    assert SUFFIX2FORMAT[".plan"] == Format.TENSORRT
    assert SUFFIX2FORMAT[".onnx"] == Format.ONNX
    assert SUFFIX2FORMAT[".pt"] == Format.TORCHSCRIPT


def test_format2suffix_returns_correct_formats():
    assert FORMAT2SUFFIX[Format.TF_SAVEDMODEL] == ".savedmodel"
    assert FORMAT2SUFFIX[Format.TENSORRT] == ".plan"
    assert FORMAT2SUFFIX[Format.ONNX] == ".onnx"
    assert FORMAT2SUFFIX[Format.TORCHSCRIPT] == ".pt"


def test_framework2base_format_returns_correct_formats():
    assert FRAMEWORK2BASE_FORMAT[Framework.NONE] == Format.PYTHON
    assert FRAMEWORK2BASE_FORMAT[Framework.TORCH] == Format.TORCH
    assert FRAMEWORK2BASE_FORMAT[Framework.JAX] == Format.JAX
    assert FRAMEWORK2BASE_FORMAT[Framework.TENSORFLOW] == Format.TENSORFLOW
    assert FRAMEWORK2BASE_FORMAT[Framework.ONNX] == Format.ONNX
