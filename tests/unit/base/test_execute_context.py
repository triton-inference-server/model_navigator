# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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


import pytest

from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorUserInputError


@pytest.mark.parametrize("verbose", [False, True])
def test_execute_context_external_cmd_success(tmp_path, mocker, verbose):
    cmd_path = tmp_path / "test_command.sh"
    workspace = Workspace(tmp_path)

    mock_logger = mocker.patch("model_navigator.commands.execution_context.LOGGER")
    with ExecutionContext(workspace=workspace, cmd_path=cmd_path, verbose=verbose) as exec_ctx:
        exec_ctx.execute_cmd(["echo", "this is a test"])
    if verbose:
        assert mock_logger.info.call_count == 2  # bake_cmd + log output
    else:
        assert mock_logger.info.call_count == 1  # only _bake_cmd


def test_execute_context_external_cmd_allowed_error(tmp_path, mocker):
    cmd_path = tmp_path / "test_command.sh"
    workspace = Workspace(tmp_path)

    mock_logger = mocker.patch("model_navigator.commands.execution_context.LOGGER")
    with ExecutionContext(workspace=workspace, cmd_path=cmd_path) as exec_ctx:
        exec_ctx.execute_cmd(["---"], allow_failure=True)
    assert mock_logger.info.call_count == 2  # cmd output
    assert mock_logger.warning.call_count == 1  # process warning


def test_execute_context_external_cmd_not_allowed_error(tmp_path, mocker):
    cmd_path = tmp_path / "test_command.sh"
    workspace = Workspace(tmp_path)

    mock_logger = mocker.patch("model_navigator.commands.execution_context.LOGGER")
    with ExecutionContext(workspace=workspace, cmd_path=cmd_path) as exec_ctx:
        with pytest.raises(ModelNavigatorUserInputError):
            exec_ctx.execute_cmd(["---"], allow_failure=False)
    assert mock_logger.info.call_count == 2  # bake_cmd + log output
