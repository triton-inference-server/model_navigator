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
from model_navigator.api.config import ProfilerConfig
from model_navigator.commands.performance import Performance, Profiler
from model_navigator.commands.performance.performance import ProfilingResults
from model_navigator.exceptions import ModelNavigatorProfilingError


def test_is_measurement_stable_return_false_when_window_is_empty():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        config=profiler_config,
        results_path=MagicMock(),
    )

    assert profiler._is_measurement_stable([]) is False


def test_is_measurement_stable_return_false_when_window_size_less_than_count():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        config=profiler_config,
        results_path=MagicMock(),
    )

    batch_size = 1
    measurements = [25, 24, 23]
    windows = [
        ProfilingResults.from_measurements(measurements, batch_size),
        ProfilingResults.from_measurements(measurements, batch_size),
    ]

    assert profiler._is_measurement_stable(windows) is False


def test_is_measurement_stable_return_false_when_avg_latencies_are_out_of_stability_range():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        config=profiler_config,
        results_path=MagicMock(),
    )

    batch_size = 1
    windows = [
        ProfilingResults.from_measurements([250, 220, 200], batch_size),
        ProfilingResults.from_measurements([200, 150, 100], batch_size),
        ProfilingResults.from_measurements([50, 49, 47], batch_size),
    ]

    assert bool(profiler._is_measurement_stable(windows)) is False


def test_is_measurement_stable_return_true_when_avg_latencies_are_in_stability_range():
    profiler_config = ProfilerConfig(batch_sizes=[1])
    profiler = Profiler(
        config=profiler_config,
        results_path=MagicMock(),
    )

    batch_size = 1
    windows = [
        ProfilingResults.from_measurements([250, 220, 200], batch_size),
        ProfilingResults.from_measurements([200, 150, 100], batch_size),
        ProfilingResults.from_measurements([52, 52, 51], batch_size),
        ProfilingResults.from_measurements([50, 49, 48], batch_size),
        ProfilingResults.from_measurements([52, 49, 47], batch_size),
    ]

    assert bool(profiler._is_measurement_stable(windows)) is True


def test_profiler_run_return_batch_sizes_upto_4_when_batch_size_4_saturates_throughput(mocker):
    mocker.patch("model_navigator.utils.dataloader.expand_sample", return_value=MagicMock())
    mocker.patch(
        "model_navigator.commands.performance.Profiler._run_measurement",
        side_effect=[
            ProfilingResults.from_measurements([10, 10, 10], 1),
            ProfilingResults.from_measurements([15, 15, 15], 2),
            ProfilingResults.from_measurements([30, 30, 30], 4),
            ProfilingResults.from_measurements([30, 30, 30], 8),
        ],
    )

    profiler_config = ProfilerConfig()
    with tempfile.NamedTemporaryFile() as temp:
        profiler = Profiler(
            config=profiler_config,
            results_path=pathlib.Path(temp.name),
        )

    results = profiler.run(runner=MagicMock(), profiling_sample=MagicMock())

    assert results[-1].batch_size == 4


def test_performance_command_returns_status_ok_when_profiling_results_found_and_profiler_exit_status_0(mocker):
    mocker.patch("subprocess.Popen.poll", return_value=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        workspace.mkdir()

        model_file = workspace / "model.pt"
        model_file.touch()

        with tempfile.NamedTemporaryFile() as tmpfile:

            mock = MagicMock()
            mock.__enter__.return_value.name = tmpfile.name
            mocker.patch("tempfile.NamedTemporaryFile", return_value=mock)
            with jsonlines.open(tmpfile.name, "w") as f:
                f.write(ProfilingResults.from_measurements([1.5], batch_size=1).to_dict())

            command_output = Performance().run(
                workspace=workspace,
                path=model_file,
                format=nav.Format.TORCHSCRIPT,
                profiler_config=ProfilerConfig(),
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

        with tempfile.NamedTemporaryFile() as tmpfile:

            mock_tempfile = MagicMock()
            mock_tempfile.__enter__.return_value.name = tmpfile.name
            mocker.patch("tempfile.NamedTemporaryFile", return_value=mock_tempfile)
            with jsonlines.open(tmpfile.name, "w") as f:
                f.write(ProfilingResults.from_measurements([1.5], batch_size=1).to_dict())

            command_output = Performance().run(
                workspace=workspace,
                path=model_file,
                format=nav.Format.TORCHSCRIPT,
                profiler_config=ProfilerConfig(),
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

        with pytest.raises(ModelNavigatorProfilingError):
            Performance().run(
                workspace=workspace,
                path=model_file,
                format=nav.Format.TORCHSCRIPT,
                profiler_config=ProfilerConfig(),
                input_metadata=MagicMock(),
                output_metadata=MagicMock(),
                batch_dim=0,
                verbose=True,
                runner_cls=MagicMock(),
            )
