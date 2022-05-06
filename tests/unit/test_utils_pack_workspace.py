# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from tempfile import TemporaryDirectory

import pytest

from model_navigator.configurator import TritonConfiguratorResult
from model_navigator.converter import ConversionConfig, ConversionResult
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.model import Format, Model, ModelConfig
from model_navigator.results import ResultsStore, State, Status
from model_navigator.utils import Workspace
from model_navigator.utils.pack_workspace import pack_workspace

FILES_DIR = pathlib.Path(__file__).parent.parent.absolute() / "files"


def test_empty_workspace():
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        workspace = Workspace(temp_dir_path)
        package_path = temp_dir_path / "test.triton.nav"

        with pytest.raises(ModelNavigatorException) as error_info:
            pack_workspace(workspace, package_path=package_path, navigator_config={})

        assert error_info.value.message == "No results found for convert_model"


def test_conversion_results_only_in_workspace():
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        workspace = Workspace(temp_dir_path)
        package_path = temp_dir_path / "test.triton.nav"

        conversion_result = ConversionResult(
            status=Status(state=State.SUCCEEDED, message="Everything fine"),
            source_model_config=ModelConfig(
                model_name="test",
                model_version="1",
                model_path=FILES_DIR / "models" / "identity.onnx",
            ),
            conversion_config=ConversionConfig(target_format=Format.ONNX),
        )
        results_store = ResultsStore(workspace)
        results_store.dump("convert_model", [conversion_result])

        with pytest.raises(ModelNavigatorException) as error_info:
            pack_workspace(workspace, package_path=package_path, navigator_config={})

        assert error_info.value.message == "No results found for configure_models_on_triton"


def test_conversion_and_configurator_results_with_minimal_info_workspace():
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        workspace = Workspace(temp_dir_path)
        package_path = temp_dir_path / "test.triton.nav"

        conversion_result = ConversionResult(
            status=Status(state=State.SUCCEEDED, message="Everything fine"),
            source_model_config=ModelConfig(
                model_name="test",
                model_version="1",
                model_path=FILES_DIR / "models" / "identity.onnx",
            ),
            conversion_config=ConversionConfig(target_format=Format.ONNX),
        )
        results_store = ResultsStore(workspace)
        results_store.dump("convert_model", [conversion_result])

        configurator_result = TritonConfiguratorResult(
            status=Status(state=State.SUCCEEDED, message="Everything fine"),
            model=Model(name="test", path=FILES_DIR / "models" / "identity.onnx"),
            model_config_name="model.config",
            model_config_path=FILES_DIR / "model-store" / "model_config1",
        )
        results_store = ResultsStore(workspace)
        results_store.dump("configure_models_on_triton", [configurator_result])

        with pytest.raises(FileNotFoundError) as error_info:
            pack_workspace(workspace, package_path=package_path, navigator_config={})

        assert "No such file or directory" in str(error_info.value)
        assert "analyzer/results" in str(error_info.value)


def test_results_checkpoints_and_models_store_exists_in_workspace():
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        workspace = Workspace(temp_dir_path)
        package_path = temp_dir_path / "test.triton.nav"

        conversion_result = ConversionResult(
            status=Status(state=State.SUCCEEDED, message="Everything fine"),
            source_model_config=ModelConfig(
                model_name="test",
                model_version="1",
                model_path=FILES_DIR / "models" / "identity.onnx",
            ),
            conversion_config=ConversionConfig(target_format=Format.ONNX),
        )
        results_store = ResultsStore(workspace)
        results_store.dump("convert_model", [conversion_result])

        configurator_result = TritonConfiguratorResult(
            status=Status(state=State.SUCCEEDED, message="Everything fine"),
            model=Model(name="test", path=FILES_DIR / "models" / "identity.onnx"),
            model_config_name="model.config",
            model_config_path=FILES_DIR / "model-store" / "model_config1",
        )
        results_store = ResultsStore(workspace)
        results_store.dump("configure_models_on_triton", [configurator_result])

        results_dir = temp_dir_path / "analyzer" / "results"
        results_dir.mkdir(parents=True)

        checkpoints_dir = temp_dir_path / "analyzer" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        model_store_dir = temp_dir_path / "analyzer" / "model-store"
        model_store_dir.mkdir(parents=True)

        pack_workspace(workspace, package_path=package_path, navigator_config={})

        assert package_path.is_file() is True
