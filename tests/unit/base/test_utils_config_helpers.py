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

from model_navigator.api.config import (
    DeviceKind,
    Format,
    OptimizationProfile,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TensorRTProfile,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import (
    ONNXConfig,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TorchModelConfig,
    TorchTensorRTConfig,
)
from model_navigator.core.constants import DEFAULT_MAX_WORKSPACE_SIZE
from model_navigator.frameworks import Framework
from model_navigator.utils.config_helpers import _do_run_max_batch_size_search, do_find_device_max_batch_size


def test__do_run_max_batch_size_search_return_false_when_batch_dim_is_none():

    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
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
        optimization_level=None,
        compatibility_level=None,
    )
    assert _do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test__do_run_max_batch_size_search_return_false_when_tensorrt_model_config_and_trt_profile_set():

    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(),
        target_device=DeviceKind.CUDA,
    )

    model_cfg = TensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profiles=[TensorRTProfile().add("x", (1,), (2,), (3,))],
        optimization_level=None,
        compatibility_level=None,
    )
    assert _do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test__do_run_max_batch_size_search_return_false_when_tftrt_model_config_and_trt_profile_set():

    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(),
        target_device=DeviceKind.CUDA,
    )

    model_cfg = TensorFlowTensorRTConfig(
        precision=TensorRTPrecision.FP16,
        minimum_segment_size=3,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profiles=[TensorRTProfile().add("x", (1,), (2,), (3,))],
    )
    assert _do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test__do_run_max_batch_size_search_return_false_when_torchtrt_model_config_and_trt_profile_set():

    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(),
        target_device=DeviceKind.CUDA,
    )

    model_cfg = TorchTensorRTConfig(
        precision=TensorRTPrecision.FP16,
        precision_mode=TensorRTPrecisionMode.HIERARCHY,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        trt_profiles=[TensorRTProfile().add("x", (1,), (2,), (3,))],
    )
    assert _do_run_max_batch_size_search(config=config, model_cfg=model_cfg) is False


def test_do_find_device_max_batch_size_return_false_when_no_cuda_device():
    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
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
                optimization_level=None,
                compatibility_level=None,
            )
        ],
    }

    assert do_find_device_max_batch_size(config, models_config) is False


def test_do_find_device_max_batch_size_return_false_when_no_adaptive_formats():

    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.ONNX, Format.TORCH),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.ONNX: [ONNXConfig(opset=17, dynamic_axes={}, graph_surgeon_optimization=True)],
        Format.TORCH: [TorchModelConfig()],
    }

    assert do_find_device_max_batch_size(config, models_config) is False


def test_do_find_device_max_batch_size_return_false_when_no_adaptive_conversion_needed():

    config = CommonConfig(
        framework=Framework.TORCH,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
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
                trt_profiles=[TensorRTProfile().add("x", (1,), (2,), (3,))],
                optimization_level=None,
                compatibility_level=None,
            )
        ],
    }

    assert do_find_device_max_batch_size(config, models_config) is False
