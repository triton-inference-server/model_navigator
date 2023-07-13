# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Tests for DeleteModel command."""
import pathlib
import tempfile

import pytest

from model_navigator.commands.delete_model import DeleteModel
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorRuntimeError


def test_delete_model_raise_error_when_file_not_exists():
    command = DeleteModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        workspace.mkdir()

        model_file = workspace / "model"

        with pytest.raises(ModelNavigatorRuntimeError):
            command.run(workspace=Workspace(workspace), path=model_file.relative_to(workspace))

        assert model_file.exists() is False


def test_delete_model_remove_item_when_model_is_file():
    command = DeleteModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        workspace.mkdir()

        model_file = workspace / "model.file"
        model_file.touch()

        command.run(workspace=Workspace(workspace), path=model_file.relative_to(workspace))

        assert model_file.exists() is False


def test_delete_model_remove_item_when_model_is_catalog():
    command = DeleteModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        workspace.mkdir()

        model_file = workspace / "model"
        model_file.mkdir()

        command.run(workspace=Workspace(workspace), path=model_file.relative_to(workspace))

        assert model_file.exists() is False
