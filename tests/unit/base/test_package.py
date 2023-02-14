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
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import model_navigator as nav
from model_navigator.exceptions import ModelNavigatorNotFoundError
from model_navigator.runners.onnx import OnnxrtCUDARunner, OnnxrtTensorRTRunner
from tests.unit.base.mocks.packages import (
    empty_package,
    onnx_package_with_cuda_runner,
    onnx_package_with_tensorrt_runner,
    tensorflow_package_with_optimal_model_tensorflow_tensorrt_and_dummy_navigator_log_dummy_status_file,
    tensorflow_package_with_tensorflow_tensorrt,
)


def test_get_runner_returns_min_latency_max_throughput_runner_when_default_strategy():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"

        package = onnx_package_with_tensorrt_runner(workspace)
        runner = package.get_runner()

        assert isinstance(runner, OnnxrtTensorRTRunner)
        assert runner.model.as_posix().endswith("onnx/model.onnx")


def test_get_runner_raises_missing_source_model_ex_when_missing_source_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"
        package = empty_package(workspace)

        with pytest.raises(ModelNavigatorNotFoundError):
            package.get_runner()


def test_get_runner_returns_source_runner_when_source_runner_is_best_and_source_model_is_loaded():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"
        package = onnx_package_with_cuda_runner(workspace)

        package.load_source_model(MagicMock())
        runner = package.get_runner()

        assert isinstance(runner, OnnxrtCUDARunner)


def test_package_get_models_paths_to_save_returns_expected_model_paths():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        package = tensorflow_package_with_tensorflow_tensorrt(workspace_path)
        model_paths_to_save = package._get_models_paths_to_save(workspace_path)

        assert len(model_paths_to_save) == 2

        expected_relative_paths = [
            "tf-savedmodel/model.savedmodel",
            "tf-trt-fp16/model.savedmodel",
        ]
        for path in model_paths_to_save:
            relative_path = os.path.relpath(path, workspace_path / ".")
            assert relative_path in expected_relative_paths


def test_make_zip_saves_excpected_files_to_archive_with_tensorflow_tensorrt_optimal_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        package = tensorflow_package_with_optimal_model_tensorflow_tensorrt_and_dummy_navigator_log_dummy_status_file(
            workspace_path
        )

        zip_path = pathlib.Path(tmp_dir) / "archive.zip"
        model_dir_to_save = [
            workspace_path / "tf-savedmodel",
            workspace_path / "tf-trt-fp16",
        ]
        files_to_save = [workspace_path / "status.yaml", workspace_path / "navigator.log"]

        package._make_zip(
            zip_path=zip_path, workspace=workspace_path, dirs_to_save=model_dir_to_save, files_to_save=files_to_save
        )

        expected_archive_content = [
            "tf-savedmodel/model.savedmodel",
            "tf-trt-fp16/model.savedmodel",
            "status.yaml",
            "navigator.log",
        ]
        with zipfile.ZipFile(zip_path) as zf:
            assert len(zf.namelist()) == 4
            for filename in zf.namelist():
                assert filename in expected_archive_content


def test_package_save_saves_expected_files_with_tensorflow_tensorrt_optimal_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        package = tensorflow_package_with_optimal_model_tensorflow_tensorrt_and_dummy_navigator_log_dummy_status_file(
            workspace_path
        )

        package_path = pathlib.Path(tmp_dir) / "nav_package.nav"
        nav.package.save(package=package, path=package_path)

        expected_archive_content = [
            "tf-savedmodel/model.savedmodel",
            "tf-trt-fp16/model.savedmodel",
            "status.yaml",
            "navigator.log",
        ]
        with zipfile.ZipFile(package_path) as zf:
            assert len(zf.namelist()) == 4
            for filename in zf.namelist():
                assert filename in expected_archive_content
