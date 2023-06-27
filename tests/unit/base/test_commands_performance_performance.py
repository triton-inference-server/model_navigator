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
import pathlib
import tempfile
from unittest.mock import MagicMock

import pytest
from jsonlines import jsonlines

import model_navigator as nav
from model_navigator.api.config import OptimizationProfile
from model_navigator.commands.performance.performance import Performance
from model_navigator.commands.performance.profiler import ProfilingResults
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorProfilingError


def test_performance_command_returns_status_ok_when_profiling_results_found_and_profiler_exit_status_0(mocker):
    mocker.patch("subprocess.Popen.poll", return_value=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        workspace.mkdir()

        model_file = workspace / "model.pt"
        model_file.touch()

        sample_file = workspace / "model_input" / "profiling" / "1.npz"
        sample_file.parent.mkdir(parents=True)
        sample_file.touch()

        with tempfile.NamedTemporaryFile() as tmpfile:

            mock = MagicMock()
            mock.__enter__.return_value.name = tmpfile.name
            mocker.patch("tempfile.NamedTemporaryFile", return_value=mock)
            with jsonlines.open(tmpfile.name, "w") as f:
                f.write(ProfilingResults.from_measurements([1.5], batch_size=1, sample_id=0).to_dict())

            command_output = Performance().run(
                workspace=Workspace(workspace),
                path=model_file,
                format=nav.Format.TORCHSCRIPT,
                optimization_profile=OptimizationProfile(),
                input_metadata=MagicMock(),
                output_metadata=MagicMock(),
                batch_dim=0,
                verbose=True,
                runner_cls=MagicMock(),
            )
    assert command_output.status == nav.CommandStatus.OK


def test_performance_command_returns_status_ok_when_profiling_results_found_and_profiler_exit_status_1(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        workspace.mkdir()

        model_file = workspace / "model.pt"
        model_file.touch()

        sample_file = workspace / "model_input" / "profiling" / "1.npz"
        sample_file.parent.mkdir(parents=True)
        sample_file.touch()

        with tempfile.NamedTemporaryFile() as tmpfile:

            mock_tempfile = MagicMock()
            mock_tempfile.__enter__.return_value.name = tmpfile.name
            mocker.patch("tempfile.NamedTemporaryFile", return_value=mock_tempfile)
            with jsonlines.open(tmpfile.name, "w") as f:
                f.write(ProfilingResults.from_measurements([1.5], batch_size=1, sample_id=0).to_dict())

            command_output = Performance().run(
                workspace=Workspace(workspace),
                path=model_file,
                format=nav.Format.TORCHSCRIPT,
                optimization_profile=OptimizationProfile(),
                input_metadata=MagicMock(),
                output_metadata=MagicMock(),
                batch_dim=0,
                verbose=True,
                runner_cls=MagicMock(),
            )
    assert command_output.status == nav.CommandStatus.OK


def test_performance_command_raises_error_when_no_profiling_results_and_profiler_exit_status_0(mocker):
    mocker.patch("subprocess.Popen.poll", return_value=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        workspace.mkdir()

        model_file = workspace / "model.pt"
        model_file.touch()

        sample_file = workspace / "model_input" / "profiling" / "1.npz"
        sample_file.parent.mkdir(parents=True)
        sample_file.touch()

        with pytest.raises(ModelNavigatorProfilingError):
            Performance().run(
                workspace=Workspace(workspace),
                path=model_file,
                format=nav.Format.TORCHSCRIPT,
                optimization_profile=OptimizationProfile(),
                input_metadata=MagicMock(),
                output_metadata=MagicMock(),
                batch_dim=0,
                verbose=True,
                runner_cls=MagicMock(),
            )
