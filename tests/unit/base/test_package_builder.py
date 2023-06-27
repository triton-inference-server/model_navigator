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

from model_navigator.package.builder import PackageBuilder
from tests.unit.base.mocks.packages import (
    tensorflow_package_with_optimal_model_tensorflow_tensorrt_and_dummy_navigator_log_dummy_status_file,
    tensorflow_package_with_tensorflow_tensorrt,
)


def test_get_models_paths_to_save_returns_expected_model_paths():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "workspace"
        package = tensorflow_package_with_tensorflow_tensorrt(workspace)

        builder = PackageBuilder()
        model_paths_to_save = builder._get_models_paths_to_save(package)

        assert len(model_paths_to_save) == 2

        expected_relative_paths = [
            "tf-savedmodel/model.savedmodel",
            "tf-trt-fp16/model.savedmodel",
        ]
        for path in model_paths_to_save:
            relative_path = os.path.relpath(path, workspace / ".")
            assert relative_path in expected_relative_paths


def test_make_zip_saves_expected_files_to_archive_with_tensorflow_tensorrt_optimal_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "workspace"
        tensorflow_package_with_optimal_model_tensorflow_tensorrt_and_dummy_navigator_log_dummy_status_file(workspace)

        zip_path = pathlib.Path(tmp_dir) / "archive.zip"
        model_dir_to_save = [
            workspace / "tf-savedmodel",
            workspace / "tf-trt-fp16",
        ]
        files_to_save = [workspace / "status.yaml", workspace / "navigator.log"]

        builder = PackageBuilder()
        builder._make_zip(
            zip_path=zip_path,
            workspace=workspace,
            dirs_to_save=model_dir_to_save,
            files_to_save=files_to_save,
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


def test_save_store_expected_files_with_tensorflow_tensorrt_optimal_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "workspace"
        package = tensorflow_package_with_optimal_model_tensorflow_tensorrt_and_dummy_navigator_log_dummy_status_file(
            workspace
        )

        package_path = pathlib.Path(tmp_dir) / "nav_package.nav"
        builder = PackageBuilder()
        builder.save(package=package, path=package_path)

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
