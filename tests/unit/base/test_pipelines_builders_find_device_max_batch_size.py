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

from model_navigator.api.config import (
    DeviceKind,
    Format,
    JitType,
    ProfilerConfig,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TensorRTProfile,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import (
    ONNXConfig,
    TensorFlowSavedModelConfig,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TorchModelConfig,
    TorchScriptConfig,
    TorchTensorRTConfig,
)
from model_navigator.core.constants import DEFAULT_MAX_WORKSPACE_SIZE
from model_navigator.frameworks import Framework
from model_navigator.pipelines.builders.find_device_max_batch_size import (
    do_run_max_batch_size_search,
    find_device_max_batch_size_builder,
)
from model_navigator.runners.onnx import OnnxrtCUDARunner
from model_navigator.runners.tensorflow import TensorFlowSavedModelCUDARunner
from model_navigator.runners.torch import TorchScriptCUDARunner


def test_do_run_max_batch_size_search_return_false_when_batch_dim_is_none():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(),
        target_device=DeviceKind.CUDA,
        batch_dim=None,
    )

    model_cfg = TensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profile=TensorRTProfile(),
    )
    assert do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test_do_run_max_batch_size_search_return_false_when_tensorrt_model_config_and_trt_profile_set():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(),
        target_device=DeviceKind.CUDA,
    )

    model_cfg = TensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profile=TensorRTProfile().add("x", (1,), (2,), (3,)),
    )
    assert do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test_do_run_max_batch_size_search_return_false_when_tftrt_model_config_and_trt_profile_set():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(),
        target_device=DeviceKind.CUDA,
    )

    model_cfg = TensorFlowTensorRTConfig(
        precision=TensorRTPrecision.FP16,
        minimum_segment_size=3,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profile=TensorRTProfile().add("x", (1,), (2,), (3,)),
    )
    assert do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test_do_run_max_batch_size_search_return_false_when_torchtrt_model_config_and_trt_profile_set():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(),
        target_device=DeviceKind.CUDA,
    )

    model_cfg = TorchTensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profile=TensorRTProfile().add("x", (1,), (2,), (3,)),
    )
    assert do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test_find_device_max_batch_size_builder_return_empty_execution_unit_when_no_cuda_device():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CPU,
    )

    models_config = {
        Format.TENSORRT: [
            TensorRTConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                trt_profile=TensorRTProfile(),
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)

    assert len(pipeline.execution_units) == 0


def test_find_device_max_batch_size_builder_return_empty_execution_unit_when_no_adaptive_formats():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.ONNX, Format.TORCH),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.ONNX: [ONNXConfig(opset=17, dynamic_axes={})],
        Format.TORCH: [TorchModelConfig()],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 0


def test_find_device_max_batch_size_builder_return_empty_execution_unit_when_no_adaptive_conversion_needed():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.TENSORRT: [
            TensorRTConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                trt_profile=TensorRTProfile().add("x", (1,), (2,), (3,)),
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 0


def test_find_device_max_batch_size_builder_return_execution_unit_when_torch_framework_is_used():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.TORCHSCRIPT: [
            TorchScriptConfig(
                jit_type=JitType.TRACE,
            )
        ],
        Format.ONNX: [ONNXConfig(opset=17, dynamic_axes={})],
        Format.TENSORRT: [
            TensorRTConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                trt_profile=TensorRTProfile(),
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 2

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == TorchScriptCUDARunner

    configuration = execution_unit.kwargs["configurations"][1]
    assert configuration.runner_cls == OnnxrtCUDARunner


def test_find_device_max_batch_size_builder_return_execution_unit_when_tensorflow_framework_is_used():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.TENSORFLOW,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.TF_SAVEDMODEL: [TensorFlowSavedModelConfig(enable_xla=False, jit_compile=False)],
        Format.ONNX: [ONNXConfig(opset=17, dynamic_axes={})],
        Format.TENSORRT: [
            TensorRTConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                trt_profile=TensorRTProfile(),
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 1

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == TensorFlowSavedModelCUDARunner


def test_find_device_max_batch_size_builder_return_execution_unit_when_jax_framework_is_used():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.JAX,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.TF_SAVEDMODEL: [TensorFlowSavedModelConfig(enable_xla=True, jit_compile=True)],
        Format.ONNX: [ONNXConfig(opset=17, dynamic_axes={})],
        Format.TENSORRT: [
            TensorRTConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                trt_profile=TensorRTProfile(),
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 1

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == TensorFlowSavedModelCUDARunner


def test_find_device_max_batch_size_builder_return_execution_unit_when_onnx_framework_is_used():
    workspace = pathlib.Path("tmp/navigator_workspace")
    config = CommonConfig(
        framework=Framework.ONNX,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        workspace=workspace,
        model=None,
        profiler_config=ProfilerConfig(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.TENSORRT,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.ONNX: [ONNXConfig(opset=17, dynamic_axes={})],
        Format.TENSORRT: [
            TensorRTConfig(
                precision=TensorRTPrecision.FP16,
                precision_mode=TensorRTPrecisionMode.HIERARCHY,
                max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                trt_profile=TensorRTProfile(),
            )
        ],
    }

    pipeline = find_device_max_batch_size_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1

    execution_unit = pipeline.execution_units[0]
    assert len(execution_unit.kwargs["configurations"]) == 1

    configuration = execution_unit.kwargs["configurations"][0]
    assert configuration.runner_cls == OnnxrtCUDARunner
