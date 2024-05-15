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
import pytest

from model_navigator.commands.correctness.correctness import Tolerance
from model_navigator.commands.performance.performance import ProfilingResults
from model_navigator.configuration import (
    JitType,
    MaxThroughputAndMinLatencyStrategy,
    MaxThroughputStrategy,
    MaxThroughputWithLatencyBudgetStrategy,
    MinLatencyStrategy,
    TensorRTPrecision,
    TensorRTPrecisionMode,
)
from model_navigator.configuration.model.model_config import ONNXConfig, TensorRTConfig, TorchScriptConfig
from model_navigator.core.constants import DEFAULT_MAX_WORKSPACE_SIZE
from model_navigator.exceptions import ModelNavigatorRuntimeAnalyzerError, ModelNavigatorUserInputError
from model_navigator.package.status import CommandStatus, ModelStatus, RunnerStatus
from model_navigator.runtime_analyzer import RuntimeAnalyzer

onnx_config = ONNXConfig(opset=13, dynamic_axes=None, dynamo_export=False, graph_surgeon_optimization=True)
tensorrt_config = TensorRTConfig(
    precision=TensorRTPrecision.FP16,
    precision_mode=TensorRTPrecisionMode.HIERARCHY,
    max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
    trt_profiles=None,
    optimization_level=None,
    compatibility_level=None,
)

model_statuses1 = {
    onnx_config.key: ModelStatus(
        model_config=onnx_config,
        runners_status={
            "OnnxCUDA": RunnerStatus(
                runner_name="OnnxCUDA",
                status={
                    "Correctness": CommandStatus.OK,
                    "Performance": CommandStatus.OK,
                    "VerifyModel": CommandStatus.OK,
                },
                result={
                    "Correctness": {"per_output_tolerance": {"output__0": Tolerance(atol=0.0, rtol=0.0)}},
                    "Performance": {
                        "profiling_results": [
                            ProfilingResults(
                                sample_id=0,
                                batch_size=1,
                                avg_latency=2.0,
                                std_latency=0.0,
                                p50_latency=2.0,
                                p90_latency=2.0,
                                p95_latency=2.0,
                                p99_latency=2.0,
                                throughput=500.0,
                                avg_gpu_clock=1500.0,
                                request_count=50,
                            ),
                        ]
                    },
                },
            ),
        },
    ),
    tensorrt_config.key: ModelStatus(
        model_config=tensorrt_config,
        runners_status={
            "TensorRT": RunnerStatus(
                runner_name="TensorRT",
                status={
                    "Correctness": CommandStatus.OK,
                    "Performance": CommandStatus.OK,
                    "VerifyModel": CommandStatus.OK,
                },
                result={
                    "Correctness": {"per_output_tolerance": {"output__0": Tolerance(atol=0.0, rtol=0.0)}},
                    "Performance": {
                        "profiling_results": [
                            ProfilingResults(
                                sample_id=0,
                                batch_size=1,
                                avg_latency=1.0,
                                std_latency=0.0,
                                p50_latency=1.0,
                                p90_latency=1.0,
                                p95_latency=1.0,
                                p99_latency=1.0,
                                throughput=1000.0,
                                avg_gpu_clock=1500.0,
                                request_count=50,
                            ),
                        ]
                    },
                },
            ),
        },
    ),
}

model_statuses2 = {
    onnx_config.key: ModelStatus(
        model_config=onnx_config,
        runners_status={
            "OnnxTensorRT": RunnerStatus(
                runner_name="OnnxTensorRT",
                status={
                    "Correctness": CommandStatus.OK,
                    "Performance": CommandStatus.OK,
                    "VerifyModel": CommandStatus.OK,
                },
                result={
                    "Correctness": {"per_output_tolerance": {"output__0": Tolerance(atol=0.0, rtol=0.0)}},
                    "Performance": {
                        "profiling_results": [
                            ProfilingResults(
                                sample_id=0,
                                batch_size=1,
                                avg_latency=2.0,
                                std_latency=0.0,
                                p50_latency=2.0,
                                p90_latency=2.0,
                                p95_latency=2.0,
                                p99_latency=2.0,
                                throughput=1000.0,
                                avg_gpu_clock=1500.0,
                                request_count=50,
                            ),
                        ]
                    },
                },
            ),
            "OnnxCUDA": RunnerStatus(
                runner_name="OnnxCUDA",
                status={
                    "Correctness": CommandStatus.OK,
                    "Performance": CommandStatus.OK,
                    "VerifyModel": CommandStatus.OK,
                },
                result={
                    "Correctness": {"per_output_tolerance": {"output__0": Tolerance(atol=0.0, rtol=0.0)}},
                    "Performance": {
                        "profiling_results": [
                            ProfilingResults(
                                sample_id=0,
                                batch_size=1,
                                avg_latency=1.0,
                                std_latency=0.0,
                                p50_latency=1.0,
                                p90_latency=1.0,
                                p95_latency=1.0,
                                p99_latency=1.0,
                                throughput=100.0,
                                avg_gpu_clock=1500.0,
                                request_count=50,
                            ),
                        ]
                    },
                },
            ),
        },
    ),
}

torchscript_config = TorchScriptConfig(jit_type=JitType.TRACE, strict=True, inference_mode=True, autocast=False)
model_statuses3 = {
    torchscript_config.key: ModelStatus(
        model_config=torchscript_config,
        runners_status={
            "TorchScriptCUDA": RunnerStatus(
                runner_name="TorchScriptCUDA",
                status={
                    "Correctness": CommandStatus.OK,
                    "Performance": CommandStatus.OK,
                    "VerifyModel": CommandStatus.OK,
                },
                result={
                    "Correctness": {"per_output_tolerance": {"output__0": Tolerance(atol=0.0, rtol=0.0)}},
                    "Performance": {
                        "profiling_results": [
                            ProfilingResults(
                                sample_id=0,
                                batch_size=1,
                                avg_latency=1.0,
                                std_latency=0.0,
                                p50_latency=1.0,
                                p90_latency=1.0,
                                p95_latency=1.0,
                                p99_latency=1.0,
                                throughput=1000.0,
                                avg_gpu_clock=1500.0,
                                request_count=50,
                            ),
                        ]
                    },
                },
            ),
            "TorchScriptCPU": RunnerStatus(
                runner_name="TorchScriptCPU",
                status={
                    "Correctness": CommandStatus.OK,
                    "Performance": CommandStatus.OK,
                    "VerifyModel": CommandStatus.OK,
                },
                result={
                    "Correctness": {"per_output_tolerance": {"output__0": Tolerance(atol=0.0, rtol=0.0)}},
                    "Performance": {
                        "profiling_results": [
                            ProfilingResults(
                                sample_id=0,
                                batch_size=1,
                                avg_latency=2.0,
                                std_latency=0.0,
                                p50_latency=2.0,
                                p90_latency=2.0,
                                p95_latency=2.0,
                                p99_latency=2.0,
                                throughput=500.0,
                                avg_gpu_clock=1500.0,
                                request_count=50,
                            ),
                        ]
                    },
                },
            ),
        },
    ),
}


def test_get_runtime_raise_error_when_unsupported_strategy_provided():
    with pytest.raises(ModelNavigatorUserInputError, match="Unsupported strategy provided:"):
        RuntimeAnalyzer.get_runtime(
            model_statuses1,
            strategy=object(),  # pytype: disable=wrong-arg-types
        )


def test_get_runtime_returns_min_latency_runner_when_strategy_is_min_latency_and_different_model_formats():
    runtime_result = RuntimeAnalyzer.get_runtime(
        model_statuses1,
        strategy=MinLatencyStrategy(),
    )

    assert isinstance(runtime_result.model_status.model_config, TensorRTConfig)
    assert runtime_result.runner_status.runner_name == "TensorRT"


def test_get_runtime_returns_min_latency_runner_when_strategy_is_min_latency_and_different_runners():
    runtime_result = RuntimeAnalyzer.get_runtime(
        model_statuses2,
        strategy=MinLatencyStrategy(),
    )

    assert isinstance(runtime_result.model_status.model_config, ONNXConfig)
    assert runtime_result.runner_status.runner_name == "OnnxCUDA"


def test_get_runtime_returns_max_throughput_runner_when_strategy_is_max_throughput_and_different_model_formats():
    runtime_result = RuntimeAnalyzer.get_runtime(
        model_statuses1,
        strategy=MaxThroughputStrategy(),
    )

    assert isinstance(runtime_result.model_status.model_config, TensorRTConfig)
    assert runtime_result.runner_status.runner_name == "TensorRT"


def test_get_runtime_returns_max_throughput_runner_when_strategy_is_max_throughput_and_different_runners():
    runtime_result = RuntimeAnalyzer.get_runtime(
        model_statuses2,
        strategy=MaxThroughputStrategy(),
    )

    assert isinstance(runtime_result.model_status.model_config, ONNXConfig)
    assert runtime_result.runner_status.runner_name == "OnnxTensorRT"


def test_get_runtime_raises_runtime_analyzer_error_when_no_runtime_satisfies_strategy():
    with pytest.raises(ModelNavigatorRuntimeAnalyzerError):
        RuntimeAnalyzer.get_runtime(
            model_statuses2,
            strategy=MaxThroughputAndMinLatencyStrategy(),
        )


def test_get_runtime_returns_min_latency_max_throughput_runner_when_min_latency_max_throughput_runner_exist():
    runtime_result = RuntimeAnalyzer.get_runtime(
        model_statuses1,
        strategy=MaxThroughputAndMinLatencyStrategy(),
    )
    assert isinstance(runtime_result.model_status.model_config, TensorRTConfig)
    assert runtime_result.runner_status.runner_name == "TensorRT"


def test_get_runtime_returns_max_thr_within_lat_budget_runner_when_strategy_is_max_throughput_with_lat_budget():
    runtime_result = RuntimeAnalyzer.get_runtime(
        model_statuses2,
        strategy=MaxThroughputWithLatencyBudgetStrategy(latency_budget=1.25),
    )
    assert isinstance(runtime_result.model_status.model_config, ONNXConfig)
    assert runtime_result.runner_status.runner_name == "OnnxCUDA"


def test_get_runtime_raise_error_when_no_result_found_with_provided_constraints():
    with pytest.raises(ModelNavigatorRuntimeAnalyzerError, match="No matching results found."):
        RuntimeAnalyzer.get_runtime(
            model_statuses2,
            strategy=MaxThroughputWithLatencyBudgetStrategy(latency_budget=0.25),
        )


def test_get_runtime_returns_min_latency_runner_when_strategy_is_min_latency_and_missing_framework():
    runtime_result = RuntimeAnalyzer.get_runtime(
        model_statuses3,
        strategy=MinLatencyStrategy(),
    )

    assert isinstance(runtime_result.model_status.model_config, TorchScriptConfig)
    assert runtime_result.runner_status.runner_name == "TorchScriptCUDA"
