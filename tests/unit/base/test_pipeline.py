# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
from unittest.mock import MagicMock

from model_navigator.commands.base import CommandOutput, CommandStatus
from model_navigator.pipelines.pipeline import Pipeline
from model_navigator.reporting.events import NavigatorEvent
from tests.unit.base.mocks.fixtures import mock_event_emitter  # noqa: F401


def test_pipeline_run_emits_events(mock_event_emitter):  # noqa: F811
    # given
    mock_exec_unit = MagicMock()
    mock_exec_unit.model_config = None
    mock_exec_unit.runner_cls = None
    mock_command = MagicMock()
    mock_exec_unit.command.name = "test_command"
    mock_exec_unit.command.return_value = mock_command
    mock_command.run.return_value = CommandOutput(CommandStatus.OK)
    pipeline = Pipeline("test_pipeline", execution_units=[mock_exec_unit])
    pipeline.event_emitter = mock_event_emitter
    mock_config = MagicMock()
    mock_config.debug = False
    # when
    pipeline.run(workspace=MagicMock(), config=mock_config, context=MagicMock())
    # then
    events = mock_event_emitter.history
    assert len(events) == 4
    assert events[0] == (NavigatorEvent.PIPELINE_STARTED, (), {"name": "test_pipeline"})
    assert events[1] == (
        NavigatorEvent.COMMAND_STARTED,
        (),
        {
            "command": "test_command",
            "config_key": None,
            "runner_name": None,
        },
    )
    assert events[2] == (NavigatorEvent.COMMAND_FINISHED, (), {"status": CommandStatus.OK})
    assert events[3] == (NavigatorEvent.PIPELINE_FINISHED, (), {})
