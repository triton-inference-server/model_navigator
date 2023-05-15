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

import numpy as np

from model_navigator.api.config import Format, JitType, TensorRTPrecision, TensorRTPrecisionMode, TensorRTProfile
from model_navigator.commands.correctness.correctness import Tolerance, TolerancePerOutputName
from model_navigator.commands.performance.performance import ProfilingResults
from model_navigator.configuration.model.model_config import (
    ONNXConfig,
    TensorFlowSavedModelConfig,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TorchModelConfig,
    TorchScriptConfig,
    TorchTensorRTConfig,
)
from model_navigator.core.constants import DEFAULT_MAX_WORKSPACE_SIZE, NAVIGATOR_PACKAGE_VERSION, NAVIGATOR_VERSION
from model_navigator.core.package import Package
from model_navigator.core.status import CommandStatus, ModelStatus, RunnerStatus, Status
from model_navigator.core.tensor import TensorMetadata, TensorSpec
from model_navigator.frameworks import Framework
from model_navigator.runners.base import NavigatorRunner


class CustomRunner(NavigatorRunner):
    @classmethod
    def format(cls):
        return Format.ONNX

    def infer_impl(self, feed_dict):
        return


def empty_package(workspace) -> Package:
    return Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TORCH.value,
                "target_device": "cpu",
                "runner_names": ("OnnxCUDA",),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TORCH.value,),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                ONNXConfig(opset=13, dynamic_axes=None).key: ModelStatus(
                    model_config=ONNXConfig(opset=13, dynamic_axes=None),
                    runners_status={
                        "OnnxCUDA": RunnerStatus(
                            runner_name="OnnxCUDA",
                            status={
                                "Correctness": CommandStatus.OK,
                                "Performance": CommandStatus.OK,
                                "VerifyModel": CommandStatus.OK,
                            },
                            result={
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )


def custom_runner_package(workspace) -> Package:
    onnx_config = ONNXConfig(opset=13, dynamic_axes=None)
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.ONNX.value,
                "target_device": "cpu",
                "runner_names": (CustomRunner.name(),),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.ONNX.value,),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                onnx_config.key: ModelStatus(
                    model_config=onnx_config,
                    runners_status={
                        CustomRunner.name(): RunnerStatus(
                            runner_name=CustomRunner.name(),
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def trochscript_package_without_source(workspace) -> Package:
    torchscript_config = TorchScriptConfig(jit_type=JitType.SCRIPT)
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TORCH.value,
                "target_device": "cpu",
                "runner_names": ("TorchCPU",),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TORCHSCRIPT.value,),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {},
            },
            models_status={
                torchscript_config.key: ModelStatus(
                    model_config=torchscript_config,
                    runners_status={
                        "TorchScriptCPU": RunnerStatus(
                            runner_name="TorchScriptCPU",
                            status={
                                "Correctness": CommandStatus.OK,
                                "Performance": CommandStatus.OK,
                                "VerifyModel": CommandStatus.OK,
                            },
                            result={
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )

    for model_status in package.status.models_status.values():
        (workspace / model_status.model_config.path).parent.mkdir(parents=True)
        (workspace / model_status.model_config.path).open("w").close()

    return package


def trochscript_package_with_source(workspace) -> Package:
    source_config = TorchModelConfig()
    torchscript_config = TorchScriptConfig(jit_type=JitType.SCRIPT)
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TORCH.value,
                "target_device": "cpu",
                "runner_names": ("TorchCPU",),
                "verbose": False,
                "debug": False,
                "target_formats": (
                    Format.TORCH.value,
                    Format.TORCHSCRIPT.value,
                ),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                    },
                    "Torch": {
                        "jit_type": (JitType.TRACE,),
                    },
                    "TensorRT": {
                        "trt_profile": {"input__0": {"min": (1, 3), "opt": (3, 3), "max": (4, 3)}},
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                source_config.key: ModelStatus(
                    model_config=source_config,
                    runners_status={
                        "TorchCPU": RunnerStatus(
                            runner_name="TorchCPU",
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
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
                torchscript_config.key: ModelStatus(
                    model_config=torchscript_config,
                    runners_status={
                        "TorchScriptCPU": RunnerStatus(
                            runner_name="TorchScriptCPU",
                            status={
                                "Correctness": CommandStatus.OK,
                                "Performance": CommandStatus.OK,
                                "VerifyModel": CommandStatus.OK,
                            },
                            result={
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )

    for model_status in package.status.models_status.values():
        (workspace / model_status.model_config.path).parent.mkdir(parents=True)
        (workspace / model_status.model_config.path).open("w").close()

    return package


def onnx_package_with_tensorrt_runner(workspace) -> Package:
    onnx_config = ONNXConfig(opset=13, dynamic_axes=None)
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.ONNX.value,
                "target_device": "cpu",
                "runner_names": ("OnnxCUDA", "OnnxTensorRT"),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.ONNX.value,),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
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
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                        "OnnxTensorRT": RunnerStatus(
                            runner_name="OnnxTensorRT",
                            status={
                                "Correctness": CommandStatus.OK,
                                "Performance": CommandStatus.OK,
                                "VerifyModel": CommandStatus.OK,
                            },
                            result={
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def onnx_package_with_cuda_runner(workspace) -> Package:
    onnx_config = ONNXConfig(opset=13, dynamic_axes=None)
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.ONNX.value,
                "runner_names": ("OnnxCPU", "OnnxCUDA"),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.ONNX.value,),
                "target_device": "cuda",
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                onnx_config.key: ModelStatus(
                    model_config=onnx_config,
                    runners_status={
                        "OnnxCPU": RunnerStatus(
                            runner_name="OnnxCPU",
                            status={
                                "Correctness": CommandStatus.OK,
                                "Performance": CommandStatus.OK,
                                "VerifyModel": CommandStatus.OK,
                            },
                            result={
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
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
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def onnx_package_with_cpu_runner_only(workspace) -> Package:
    onnx_config = ONNXConfig(opset=13, dynamic_axes=None)
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.ONNX.value,
                "runner_names": ("OnnxCPU",),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.ONNX.value,),
                "target_device": "cpu",
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                onnx_config.key: ModelStatus(
                    model_config=onnx_config,
                    runners_status={
                        "OnnxCPU": RunnerStatus(
                            runner_name="OnnxCPU",
                            status={
                                "Correctness": CommandStatus.OK,
                                "Performance": CommandStatus.OK,
                                "VerifyModel": CommandStatus.OK,
                            },
                            result={
                                "Correctness": {
                                    "per_output_tolerance": TolerancePerOutputName(
                                        {"output__0": Tolerance(atol=0.0, rtol=0.0)}
                                    )
                                },
                                "Performance": {
                                    "profiling_results": [
                                        ProfilingResults(
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def tensorflow_package_with_tensorflow_only(workspace) -> Package:
    tensorflow_savedmodel_config = TensorFlowSavedModelConfig(
        enable_xla=None,
        jit_compile=None,
    )
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TENSORFLOW.value,
                "runner_names": ("OnnxCPU",),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TF_SAVEDMODEL.value,),
                "target_device": "cpu",
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {},
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                tensorflow_savedmodel_config.key: ModelStatus(
                    model_config=tensorflow_savedmodel_config,
                    runners_status={
                        "TensorFlowSavedModelCPU": RunnerStatus(
                            runner_name="TensorFlowSavedModelCPU",
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
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def tensorflow_package_with_tensorflow_tensorrt(workspace) -> Package:
    tensorflow_savedmodel_config = TensorFlowSavedModelConfig(
        enable_xla=None,
        jit_compile=None,
    )
    tensorflow_tensorrt_config = TensorFlowTensorRTConfig(
        precision=TensorRTPrecision.FP16,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        minimum_segment_size=3,
        trt_profile=None,
    )

    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TENSORFLOW.value,
                "runner_names": ("OnnxCPU",),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TF_SAVEDMODEL.value, Format.TORCH_TRT.value),
                "target_device": "cpu",
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {},
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                tensorflow_savedmodel_config.key: ModelStatus(
                    model_config=tensorflow_savedmodel_config,
                    runners_status={
                        "TensorFlowSavedModelCPU": RunnerStatus(
                            runner_name="TensorFlowSavedModelCPU",
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
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
                tensorflow_tensorrt_config.key: ModelStatus(
                    model_config=tensorflow_tensorrt_config,
                    runners_status={
                        "TensorFlowTensorRT": RunnerStatus(
                            runner_name="TensorFlowTensorRT",
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def tensorflow_package_with_optimal_model_tensorflow_tensorrt_and_dummy_navigator_log_dummy_status_file(
    workspace,
) -> Package:
    tensorflow_savedmodel_config = TensorFlowSavedModelConfig(
        enable_xla=None,
        jit_compile=None,
    )
    tensorflow_tensorrt_config = TensorFlowTensorRTConfig(
        precision=TensorRTPrecision.FP16,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        minimum_segment_size=3,
        trt_profile=None,
    )

    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TENSORFLOW.value,
                "runner_names": ("OnnxCPU",),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TF_SAVEDMODEL.value, Format.TORCH_TRT.value),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {},
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                tensorflow_savedmodel_config.key: ModelStatus(
                    model_config=tensorflow_savedmodel_config,
                    runners_status={
                        "TensorFlowSavedModelCPU": RunnerStatus(
                            runner_name="TensorFlowSavedModelCPU",
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
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
                tensorflow_tensorrt_config.key: ModelStatus(
                    model_config=tensorflow_tensorrt_config,
                    runners_status={
                        "TensorFlowTensorRT": RunnerStatus(
                            runner_name="TensorFlowTensorRT",
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    with open(workspace / "status.yaml", "w") as f:
        f.write("dummy content")

    with open(workspace / "navigator.log", "w") as f:
        f.write("dummy content")

    return package


def torchscript_package_with_cpu_only(workspace) -> Package:
    torchscript_config = TorchScriptConfig(jit_type=JitType.TRACE)

    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TORCH.value,
                "target_device": "cpu",
                "runner_names": (
                    "TorchScriptCUDA",
                    "TorchScriptCPU",
                ),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TORCHSCRIPT.value,),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
                torchscript_config.key: ModelStatus(
                    model_config=torchscript_config,
                    runners_status={
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
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def torchscript_package_with_cuda(workspace) -> Package:
    torchscript_config = TorchScriptConfig(jit_type=JitType.TRACE)
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TORCH.value,
                "target_device": "cpu",
                "runner_names": (
                    "TorchScriptCUDA",
                    "TorchScriptCPU",
                ),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TORCHSCRIPT.value,),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                        "dynamic_axes": {"input__0": [0]},
                    },
                    "TensorRT": {
                        "trt_profile": {"input__0": {"min": (1, 3), "opt": (3, 3), "max": (4, 3)}},
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
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
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def torchscript_package_with_torch_tensorrt(workspace) -> Package:
    torchscript_config = TorchScriptConfig(jit_type=JitType.TRACE)
    torchtensorrt_config = TorchTensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profile=None,
    )
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TORCH.value,
                "target_device": "cpu",
                "runner_names": (
                    "TorchScriptCUDA",
                    "TorchTensorRT",
                ),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TORCHSCRIPT.value, Format.TORCH_TRT.value),
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                        "dynamic_axes": {"input__0": [0]},
                    },
                    "TensorRT": {
                        "trt_profile": {"input__0": {"min": (1, 3), "opt": (3, 3), "max": (4, 3)}},
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
                torchtensorrt_config.key: ModelStatus(
                    model_config=torchtensorrt_config,
                    runners_status={
                        "TorchTensorRT": RunnerStatus(
                            runner_name="TorchTensorRT",
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
                                            batch_size=1,
                                            avg_latency=2.0,
                                            std_latency=0.0,
                                            p50_latency=2.0,
                                            p90_latency=2.0,
                                            p95_latency=2.0,
                                            p99_latency=2.0,
                                            throughput=500.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def onnx_package(workspace) -> Package:
    onnx_config = ONNXConfig(opset=13, dynamic_axes={"input__0": [0]})
    tensorrt_config = TensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profile=None,
    )
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.ONNX.value,
                "runner_names": (
                    "TensorRT",
                    "OnnxCUDA",
                ),
                "verbose": False,
                "debug": False,
                "target_formats": (
                    Format.ONNX.value,
                    Format.TENSORRT.value,
                ),
                "target_device": "cuda",
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                        "dynamic_axes": {"input__0": [0]},
                    },
                    "TensorRT": {
                        "trt_profile": {"input__0": {"min": (1, 3), "opt": (3, 3), "max": (4, 3)}},
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package


def tensorrt_package(workspace) -> Package:
    tensorrt_config = TensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profile=None,
    )
    package = Package(
        status=Status(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid="1",
            environment={},
            config={
                "framework": Framework.TORCH.value,
                "runner_names": ("TensorRT",),
                "verbose": False,
                "debug": False,
                "target_formats": (Format.TENSORRT.value,),
                "target_device": "cuda",
                "sample_count": 1,
                "batch_dim": 0,
                "custom_configs": {
                    "Onnx": {
                        "opset": 13,
                        "dynamic_axes": {"input__0": [0]},
                    },
                    "TensorRT": {
                        "trt_profile": {"input__0": {"min": (1, 3), "opt": (3, 3), "max": (4, 3)}},
                    },
                },
                "profiler_config": {
                    "batch_sizes": [1, 32],
                },
            },
            models_status={
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
                                            batch_size=1,
                                            avg_latency=1.0,
                                            std_latency=0.0,
                                            p50_latency=1.0,
                                            p90_latency=1.0,
                                            p95_latency=1.0,
                                            p99_latency=1.0,
                                            throughput=1000.0,
                                            request_count=50,
                                        ),
                                    ]
                                },
                            },
                        ),
                    },
                ),
            },
            input_metadata=TensorMetadata({"input__0": TensorSpec("input__0", (-1, 1), np.dtype("float32"))}),
            output_metadata=TensorMetadata({"output__0": TensorSpec("output__0", (-1, 1), np.dtype("float32"))}),
            dataloader_trt_profile=TensorRTProfile(),
            dataloader_max_batch_size=2,
        ),
        workspace=workspace,
        model=None,
    )
    for models_status in package.status.models_status.values():
        (workspace / models_status.model_config.path).parent.mkdir(parents=True)
        (workspace / models_status.model_config.path).open("w").close()

    return package
