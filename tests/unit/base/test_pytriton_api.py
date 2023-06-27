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

import os
import pathlib
import tempfile

import numpy as np
import pytest

from model_navigator.api.pytriton import DynamicBatcher, PyTritonAdapter
from model_navigator.exceptions import ModelNavigatorNotFoundError
from tests.unit.base.mocks.packages import empty_package, onnx_package_with_cuda_runner


def test_pytriton_adapter_returns_valid_model_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        package = onnx_package_with_cuda_runner(workspace)
        adapter = PyTritonAdapter(package)
        model_config = adapter.config

        assert model_config.max_batch_size == 1
        assert model_config.batcher == DynamicBatcher()
        assert model_config.response_cache is False


def test_pytriton_adapter_returns_valid_inputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        package = onnx_package_with_cuda_runner(workspace)
        adapter = PyTritonAdapter(package)
        inputs = adapter.inputs

        assert len(inputs) == 1
        assert inputs[0].name == "input__0"
        assert inputs[0].dtype == np.float32
        assert inputs[0].shape == (1,)


def test_pytriton_adapter_returns_valid_outputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        package = onnx_package_with_cuda_runner(workspace)
        adapter = PyTritonAdapter(package)
        outputs = adapter.outputs

        assert len(outputs) == 1
        assert outputs[0].name == "output__0"
        assert outputs[0].dtype == np.float32
        assert outputs[0].shape == (1,)


def test_pytriton_adapter_returns_valid_batching():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        package = onnx_package_with_cuda_runner(workspace)
        adapter = PyTritonAdapter(package)
        assert adapter.batching is True


def test_pytriton_adapter_selects_valid_runner():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        package = onnx_package_with_cuda_runner(workspace)
        adapter = PyTritonAdapter(package)
        assert adapter.runner.name() == "OnnxCUDA"
        assert adapter.runner.is_active is False
        assert os.path.relpath(adapter.runner._model, tmp_dir) == "navigator_workspace/onnx/model.onnx"


def test_pytriton_adapter_with_empty_package():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        package = empty_package(workspace)
        with pytest.raises(ModelNavigatorNotFoundError):
            PyTritonAdapter(package)
