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
import pathlib
import tempfile

import numpy as np

import model_navigator as nav
from model_navigator.api.config import (
    AVAILABLE_NONE_FRAMEWORK_TARGET_FORMATS,
    AVAILABLE_TARGET_FORMATS,
    DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS,
    DEFAULT_TARGET_FORMATS,
    EXPORT_FORMATS,
    INPUT_FORMATS,
    Format,
)
from model_navigator.utils.framework import Framework
from model_navigator.utils.tensor import TensorSpec


def infer_func(input__0):
    return {"output__0": input__0}


dataloader = [np.random.rand(3, 5).astype("float32") for _ in range(10)]


def verify_func(ys_runner, ys_expected):
    for y_runner, y_expected in zip(ys_runner, ys_expected):
        if not all(
            [np.allclose(a, b, rtol=1.0e-3, atol=1.0e-3) for a, b in zip(y_runner.values(), y_expected.values())]
        ):
            return False
    return True


def test_python_package_return_valid_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = nav.python.optimize(
            model=infer_func,
            dataloader=dataloader,
            verify_func=verify_func,
            profiler_config=nav.ProfilerConfig(
                batch_sizes=[1, 2, 4],
            ),
            workspace=workspace,
        )
        runner = package.get_runner()
        assert runner is not None
        assert runner.format() == nav.Format.PYTHON
        assert runner.input_metadata == {
            "input__0": TensorSpec(name="input__0", shape=(-1, 5), dtype=np.dtype("float32"), optional=False)
        }
        assert runner.output_metadata == {
            "output__0": TensorSpec(name="output__0", shape=(-1, 5), dtype=np.dtype("float32"), optional=False)
        }

        with runner:
            input = {"input__0": dataloader[0]}
            output = runner.infer(input)
            assert output is not None
            assert output["output__0"] is not None
            assert np.array_equal(output["output__0"], input["input__0"])


def test_export_formats_return_empty_list_for_framework_none():
    assert len(EXPORT_FORMATS[Framework.NONE]) == 0


def test_input_formats_return_python_for_framework_none():
    assert INPUT_FORMATS[Framework.NONE] == Format.PYTHON


def test_available_none_framework_target_formats_contains_only_python_format():
    assert len(AVAILABLE_NONE_FRAMEWORK_TARGET_FORMATS) == 1
    assert AVAILABLE_NONE_FRAMEWORK_TARGET_FORMATS[0] == Format.PYTHON


def test_available_target_formats_contains_none_framework_target_formats():
    assert Framework.NONE in AVAILABLE_TARGET_FORMATS
    assert AVAILABLE_TARGET_FORMATS[Framework.NONE] == AVAILABLE_NONE_FRAMEWORK_TARGET_FORMATS


def test_default_target_format_return_python_format_for_none_framework():
    assert len(DEFAULT_TARGET_FORMATS[Framework.NONE]) == 1
    assert DEFAULT_TARGET_FORMATS[Framework.NONE] == DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS


def default_none_framework_target_formats_return_python_format():
    assert len(DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS) == 1
    assert DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS == (Format.PYTHON,)
