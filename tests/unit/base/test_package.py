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
from unittest.mock import MagicMock

import pytest

from model_navigator.exceptions import ModelNavigatorNotFoundError
from model_navigator.runners.onnx import OnnxrtCUDARunner, OnnxrtTensorRTRunner
from tests.unit.base.mocks.packages import (
    empty_package,
    onnx_package_with_cuda_runner,
    onnx_package_with_tensorrt_runner,
)


def test_get_runner_returns_min_latency_max_throughput_runner_when_default_strategy():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        package = onnx_package_with_tensorrt_runner(workspace)
        runner = package.get_runner()

        assert isinstance(runner, OnnxrtTensorRTRunner)
        assert runner.model.as_posix().endswith("onnx/model.onnx")


def test_get_runner_raises_missing_source_model_ex_when_missing_source_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = empty_package(workspace)

        with pytest.raises(ModelNavigatorNotFoundError):
            package.get_runner()


def test_get_runner_returns_source_runner_when_source_runner_is_best_and_source_model_is_loaded():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = onnx_package_with_cuda_runner(workspace)

        package.load_source_model(MagicMock())
        runner = package.get_runner()

        assert isinstance(runner, OnnxrtCUDARunner)
