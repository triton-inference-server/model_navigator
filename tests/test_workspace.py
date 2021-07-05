# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import tempfile
from pathlib import Path

from model_navigator.utils.workspace import Workspace


def test_workspace_exists():
    """Workspace path exists - is created"""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Workspace(temp_dir)

        assert workspace.path == Path(temp_dir)
        assert workspace.path.exists()
        assert workspace.exists()

        dummy_workspace_path = Path(temp_dir) / "dummy/workspace"
        workspace = Workspace(dummy_workspace_path)

        assert workspace.path == dummy_workspace_path
        assert workspace.path.exists()
        assert workspace.exists()


def test_workspace_empty():
    """Verifying workspace empty method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Workspace(temp_dir)

        assert workspace.path == Path(temp_dir)
        assert workspace.empty()

        _create_dummy_file(workspace)

        assert not workspace.empty()


def test_workspace_cleaning():
    """Test cleaning of workspace"""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Workspace(temp_dir)

        _create_dummy_file(workspace)

        assert not workspace.empty()

        workspace.clean()

        assert workspace.exists()
        assert workspace.empty()


def _create_dummy_file(workspace):
    dummy_path = workspace.path / "foo/bar.txt"
    dummy_path.parent.mkdir(parents=True)
    with dummy_path.open("w") as dummy_file:
        dummy_file.write("foo bar")
