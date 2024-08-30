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


from pathlib import Path, PosixPath  # noqa: F401 - required by eval
from unittest.mock import MagicMock  # noqa: F401 - required by eval

import pytest

from model_navigator.inplace.profiling import ProfilingResult  # noqa: F401 - required by eval
from model_navigator.reporting.profile.events import ProfileEvent  # noqa: I001
from model_navigator.reporting.profile.report import SimpleReport
from tests.unit.base.mocks.fixtures import mock_event_emitter  # noqa: F401
from tests.utils import get_assets_path

SCENARIO_PATH = get_assets_path() / "reports" / "profile"


def emit_events(event_emitter, filename):
    """Emits events recorded in the provided file."""
    with open(SCENARIO_PATH / filename) as fp:
        while line := fp.readline():
            event_str, kwargs_str = line.strip().split(maxsplit=1)
            event: ProfileEvent = eval(event_str)
            kwargs = eval(kwargs_str)
            event_emitter.emit(event, **kwargs)


@pytest.mark.parametrize("scenario_name", ["profile_with_results"])
def test_simple_report_with_results(scenario_name, mock_event_emitter):  # noqa: F811
    # given
    report = SimpleReport(event_emitter=mock_event_emitter, width=200)
    # when
    emit_events(mock_event_emitter, f"scenario_{scenario_name}.txt")
    # then
    result = report.console.export_text()
    expected = Path.read_text(SCENARIO_PATH / f"scenario_{scenario_name}_expected.txt")

    assert result == expected


@pytest.mark.parametrize("scenario_name", ["profile_without_results"])
def test_simple_report_without_results(scenario_name, mock_event_emitter):  # noqa: F811
    # given
    report = SimpleReport(event_emitter=mock_event_emitter, show_results=False)
    # when
    emit_events(mock_event_emitter, f"scenario_{scenario_name}.txt")
    # then
    result = report.console.export_text()
    expected = Path.read_text(SCENARIO_PATH / f"scenario_{scenario_name}_expected.txt")

    assert result == expected
