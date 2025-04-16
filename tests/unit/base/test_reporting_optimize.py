# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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


from pathlib import Path, PosixPath  # noqa: F401 - required by eval
from unittest.mock import MagicMock  # noqa: F401 - required by eval

import pytest

from model_navigator.commands.base import CommandStatus  # noqa: F401 - required by eval
from model_navigator.reporting.optimize.detailed_report import DetailedReport
from model_navigator.reporting.optimize.events import OptimizeEvent  # noqa: I001
from model_navigator.reporting.optimize.simple_report import SimpleReport
from tests.unit.base.mocks.fixtures import mock_event_emitter  # noqa: F401
from tests.utils import get_assets_path

SCENARIO_PATH = get_assets_path() / "reports" / "optimize"


def emit_events(event_emitter, filename):
    """Emits events recorded in the provided file."""
    with open(SCENARIO_PATH / filename) as fp:
        while line := fp.readline():
            event_str, kwargs_str = line.strip().split(maxsplit=1)
            event: OptimizeEvent = eval(event_str)
            kwargs = eval(kwargs_str)
            event_emitter.emit(event, **kwargs)


@pytest.mark.parametrize("scenario_name", ["inplace", "torch", "jax", "onnx"])
def test_simple_report(scenario_name, mock_event_emitter):  # noqa: F811
    # given
    report = SimpleReport(mock_event_emitter)
    report.save_report_to_workspace = MagicMock()
    # when
    emit_events(mock_event_emitter, f"scenario_{scenario_name}.txt")
    # then
    result = report.console.export_text()
    expected = Path.read_text(SCENARIO_PATH / f"scenario_{scenario_name}_expected.txt")

    assert result == expected, f"Scenario {scenario_name} failed"
    report.save_report_to_workspace.assert_called()


@pytest.mark.parametrize("scenario_name", ["inplace", "torch", "jax", "onnx"])
def test_detailed_report(scenario_name, mock_event_emitter):  # noqa: F811
    # given
    report = DetailedReport(mock_event_emitter)
    report.save_report_to_workspace = MagicMock()
    # when
    emit_events(mock_event_emitter, f"scenario_{scenario_name}.txt")
    # then
    # detailed report is similar to simple report with only table added
    # check that there is no exception
    result = report.console.export_text()
    assert len(result)  # no empty report
    report.save_report_to_workspace.assert_called()
