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

import json

from model_navigator.core.workspace import Workspace
from model_navigator.package import optimize
from model_navigator.package.package import Package
from model_navigator.package.status import Status
from tests.unit.base.mocks.packages import empty_package
from tests.unit.base.mocks.statuses import (
    status_dict_v0_1_0,
    status_dict_v0_1_2,
    status_dict_v0_1_3,
    status_dict_v0_1_4,
    status_dict_v0_2_0,
    status_dict_v0_2_1,
    status_dict_v0_2_2,
    status_dict_v0_2_3,
    status_dict_v0_3_0,
)


def test_from_dict_returns_status_when_input_is_0_1_0_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_1_0()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir(exist_ok=True)

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_1_2_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_1_2()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir(exist_ok=True)

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_1_3_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_1_3()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir(exist_ok=True)

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_1_4_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_1_4()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir()

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_2_0_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_2_0()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir()

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_2_1_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_2_1()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir()

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_2_2_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_2_2()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir()

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_2_3_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_2_3()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir()

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_from_dict_returns_status_when_input_is_0_3_0_status_dict(mocker, tmp_path):
    status_dict = status_dict_v0_3_0()
    status = Status.from_dict(status_dict)

    workspace = tmp_path / "navigator_workspace"
    workspace.mkdir()

    for model_status in status.models_status.values():
        (workspace / model_status.model_config.path.parent).mkdir()

    package = Package(status=status, workspace=Workspace(workspace))
    package.save_status_file()

    mocker.patch("model_navigator.package.optimize_pipeline")
    mocker.patch("model_navigator.package._get_builders")

    optimize(package=package)


def test_status_returns_itself_when_deserialized_and_serialized(tmp_path):
    workspace = tmp_path / "navigator_workspace"
    package = empty_package(workspace)
    status = package.status
    serialized_status = status.to_dict(parse=True)
    deserialized_status = Status.from_dict(serialized_status).to_dict(parse=True)
    assert json.dumps(serialized_status) == json.dumps(deserialized_status)
