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
"""Tests for FindMaxBatchSize command.

Note:
     Those test do not execute the profiling with search.
     The tests are checking if correct paths are executed on input arguments.
"""

import pathlib
import tempfile
from unittest.mock import MagicMock

import jsonlines
import numpy as np

from model_navigator.commands.base import CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.commands.find_max_batch_size import FindMaxBatchSize, FindMaxBatchSizeConfig
from model_navigator.configuration import Format, OptimizationProfile
from model_navigator.core.tensor import TensorMetadata, TensorSpec
from model_navigator.core.workspace import Workspace
from model_navigator.runners.onnx import OnnxrtCPURunner


def test_find_max_batch_size_return_none_when_model_not_support_batching(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"

        model_path = tmpdir / "model.onnx"
        model_path.touch()

        with mocker.patch.object(ExecutionContext, "execute_python_script"):
            result = FindMaxBatchSize().run(
                configurations=[
                    FindMaxBatchSizeConfig(
                        format=Format.ONNX,
                        model_path=model_path,
                        runner_cls=OnnxrtCPURunner,
                        reproduction_scripts_dir=pathlib.Path("onnx"),
                    )
                ],
                workspace=Workspace(workspace),
                input_metadata=TensorMetadata({
                    "input__1": TensorSpec(name="input__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                output_metadata=TensorMetadata({
                    "output__1": TensorSpec(name="output__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                optimization_profile=OptimizationProfile(),
                batch_dim=None,
                verbose=True,
            )

            assert result is not None
            assert result.status == CommandStatus.OK
            assert result.output == {"device_max_batch_size": None}
            assert ExecutionContext.execute_python_script.called is False  # pytype: disable=attribute-error


def test_find_max_batch_size_return_max_batch_when_model_support_batching(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"

        model_path = tmpdir / "model.onnx"
        model_path.touch()

        profiling_result_data = [
            {
                "batch_size": 16,
                "avg_latency": 1,
                "std_latency": 1,
                "p50_latency": 1,
                "p90_latency": 1,
                "p95_latency": 1,
                "p99_latency": 1,
                "throughput": 1,
                "avg_gpu_clock": 1,
                "request_count": 1,
            },
            {
                "batch_size": 64,
                "avg_latency": 1,
                "std_latency": 1,
                "p50_latency": 1,
                "p90_latency": 1,
                "p95_latency": 1,
                "p99_latency": 1,
                "throughput": 1,
                "avg_gpu_clock": 1,
                "request_count": 1,
            },
            {
                "batch_size": 128,
                "avg_latency": 1,
                "std_latency": 1,
                "p50_latency": 1,
                "p90_latency": 1,
                "p95_latency": 1,
                "p99_latency": 1,
                "throughput": 1,
                "avg_gpu_clock": 1,
                "request_count": 1,
            },
        ]
        results_file = tmpdir / "results.json"
        with jsonlines.open(results_file.as_posix(), "a") as f:
            for result in profiling_result_data:
                f.write(result)

        mock = MagicMock()
        mock.__enter__.return_value.name = results_file.as_posix()

        with mocker.patch.object(
            ExecutionContext,
            "execute_python_script",
        ), mocker.patch("tempfile.NamedTemporaryFile", return_value=mock):
            result = FindMaxBatchSize().run(
                configurations=[
                    FindMaxBatchSizeConfig(
                        format=Format.ONNX,
                        model_path=model_path,
                        runner_cls=OnnxrtCPURunner,
                        reproduction_scripts_dir=model_path.parent,
                    )
                ],
                workspace=Workspace(workspace),
                input_metadata=TensorMetadata({
                    "input__1": TensorSpec(name="input__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                output_metadata=TensorMetadata({
                    "output__1": TensorSpec(name="output__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                optimization_profile=OptimizationProfile(),
                batch_dim=0,
                verbose=True,
            )

            assert result is not None
            assert result.status == CommandStatus.OK
            assert result.output == {"device_max_batch_size": 128}
            assert ExecutionContext.execute_python_script.called is True  # pytype: disable=attribute-error


def test_find_max_batch_size_return_none_when_batch_dim_none_and_max_batch_size_provided_in_optimization_profile(
    mocker,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"

        model_path = tmpdir / "model.onnx"
        model_path.touch()

        with mocker.patch.object(ExecutionContext, "execute_python_script"):
            result = FindMaxBatchSize().run(
                configurations=[
                    FindMaxBatchSizeConfig(
                        format=Format.ONNX,
                        model_path=model_path,
                        runner_cls=OnnxrtCPURunner,
                        reproduction_scripts_dir=pathlib.Path("onnx"),
                    )
                ],
                workspace=Workspace(workspace),
                input_metadata=TensorMetadata({
                    "input__1": TensorSpec(name="input__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                output_metadata=TensorMetadata({
                    "output__1": TensorSpec(name="output__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                optimization_profile=OptimizationProfile(max_batch_size=16),
                batch_dim=None,
                verbose=True,
            )

            assert result is not None
            assert result.status == CommandStatus.OK
            assert result.output == {"device_max_batch_size": None}
            assert ExecutionContext.execute_python_script.called is False  # pytype: disable=attribute-error


def test_find_max_batch_size_return_max_batch_when_max_batch_size_provided_in_optimization_profile(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"

        model_path = tmpdir / "model.onnx"
        model_path.touch()

        with mocker.patch.object(ExecutionContext, "execute_python_script"):
            result = FindMaxBatchSize().run(
                configurations=[
                    FindMaxBatchSizeConfig(
                        format=Format.ONNX,
                        model_path=model_path,
                        runner_cls=OnnxrtCPURunner,
                        reproduction_scripts_dir=pathlib.Path("onnx"),
                    )
                ],
                workspace=Workspace(workspace),
                input_metadata=TensorMetadata({
                    "input__1": TensorSpec(name="input__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                output_metadata=TensorMetadata({
                    "output__1": TensorSpec(name="output__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                optimization_profile=OptimizationProfile(max_batch_size=16),
                batch_dim=[0],
                verbose=True,
            )

            assert result is not None
            assert result.status == CommandStatus.OK
            assert result.output == {"device_max_batch_size": 16}
            assert ExecutionContext.execute_python_script.called is False  # pytype: disable=attribute-error


def test_find_max_batch_size_return_max_batch_when_batch_sizes_provided_in_optimization_profile(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"

        model_path = tmpdir / "model.onnx"
        model_path.touch()

        with mocker.patch.object(ExecutionContext, "execute_python_script"):
            result = FindMaxBatchSize().run(
                configurations=[
                    FindMaxBatchSizeConfig(
                        format=Format.ONNX,
                        model_path=model_path,
                        runner_cls=OnnxrtCPURunner,
                        reproduction_scripts_dir=pathlib.Path("onnx"),
                    )
                ],
                workspace=Workspace(workspace),
                input_metadata=TensorMetadata({
                    "input__1": TensorSpec(name="input__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                output_metadata=TensorMetadata({
                    "output__1": TensorSpec(name="output__1", shape=(-1,), dtype=np.dtype("float32"))
                }),
                optimization_profile=OptimizationProfile(batch_sizes=[1, 2, 4]),
                batch_dim=[0],
                verbose=True,
            )

            assert result is not None
            assert result.status == CommandStatus.OK
            assert result.output == {"device_max_batch_size": 4}
            assert ExecutionContext.execute_python_script.called is False  # pytype: disable=attribute-error
