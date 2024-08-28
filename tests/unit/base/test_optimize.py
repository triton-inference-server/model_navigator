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
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from model_navigator import CommandStatus
from model_navigator.exceptions import ModelNavigatorRuntimeAnalyzerError
from model_navigator.package.builder import PackageBuilder
from model_navigator.package.status import ModelStatus, RunnerStatus
from model_navigator.pipelines.pipeline_manager import PipelineManager
from model_navigator.pipelines.wrappers.optimize import optimize_pipeline
from model_navigator.reporting.events import NavigatorEvent
from model_navigator.runtime_analyzer.analyzer import RuntimeAnalyzerResult
from tests.unit.base.mocks.fixtures import mock_event_emitter  # noqa: F401


def test_optimize_pipeline_emits_events_no_results(mocker, mock_event_emitter):  # noqa: F811
    # given
    empty_mock_package = MagicMock()
    empty_mock_package.get_best_runtime.side_effect = ModelNavigatorRuntimeAnalyzerError("test_exception")

    with mocker.patch.object(PipelineManager, "run"):
        with mocker.patch.object(PackageBuilder, "create", return_value=empty_mock_package):
            with mocker.patch(
                "model_navigator.pipelines.wrappers.optimize.default_event_emitter", return_value=mock_event_emitter
            ):
                # when
                optimize_pipeline(
                    workspace=Path("test_path"),
                    builders=MagicMock(),
                    config=MagicMock(),
                    model=MagicMock(),
                    package=None,
                    models_config=None,
                )

    # then
    events = mock_event_emitter.history
    assert len(events) == 4
    assert events[0] == (NavigatorEvent.WORKSPACE_INITIALIZED, (), {"path": Path("test_path")})
    assert events[1] == (NavigatorEvent.OPTIMIZATION_STARTED, (), {})
    assert events[2] == (NavigatorEvent.MODEL_NOT_OPTIMIZED_ERROR, (), {})
    assert events[3] == (NavigatorEvent.OPTIMIZATION_FINISHED, (), {})


@pytest.mark.parametrize("is_source_model", [False, True])
def test_optimize_pipeline_emits_events_correct_opt_results(is_source_model, mocker, mock_event_emitter, tmp_path):  # noqa: F811
    # given
    mock_package = MagicMock()
    model_config = MagicMock()
    model_config.key = "test_key"
    model_config.path = "opt_path"
    model_status = ModelStatus(model_config=model_config, runners_status={"test_runner": MagicMock()})
    runner_status = RunnerStatus(runner_name="test_runner", status={"Performance": CommandStatus.OK})
    runtime_analyzer_result = RuntimeAnalyzerResult(
        model_status=model_status, runner_status=runner_status, latency=1, throughput=1
    )

    mock_package.get_best_runtime.return_value = runtime_analyzer_result
    mock_package.workspace.path = tmp_path
    if is_source_model:
        expected_path = None
    else:
        expected_path = tmp_path / model_config.path
        expected_path.mkdir(exist_ok=True)

    with mocker.patch.object(PipelineManager, "run"):
        with mocker.patch.object(PackageBuilder, "create", return_value=mock_package):
            with mocker.patch(
                "model_navigator.pipelines.wrappers.optimize.default_event_emitter", return_value=mock_event_emitter
            ):
                # when
                optimize_pipeline(
                    workspace=Path("test_path"),
                    builders=MagicMock(),
                    config=MagicMock(),
                    model=MagicMock(),
                    package=None,
                    models_config=None,
                )

    # then
    events = mock_event_emitter.history
    assert len(events) == 4
    assert events[0] == (NavigatorEvent.WORKSPACE_INITIALIZED, (), {"path": Path("test_path")})
    assert events[1] == (NavigatorEvent.OPTIMIZATION_STARTED, (), {})
    assert events[2] == (
        NavigatorEvent.BEST_MODEL_PICKED,
        (),
        {
            "config_key": "test_key",
            "runner_name": "test_runner",
            "model_path": expected_path,
        },
    )
    assert events[3] == (NavigatorEvent.OPTIMIZATION_FINISHED, (), {})
