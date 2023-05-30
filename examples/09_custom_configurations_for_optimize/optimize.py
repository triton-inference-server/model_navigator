#!/usr/bin/env python3
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

from typing import Iterable

import numpy as np
import torch  # pytype: disable=import-error

import model_navigator as nav
from model_navigator.api.config import Sample


def get_model():
    """Returns a simple torch.nn.Linear model"""
    return torch.nn.Linear(5, 7).eval()


def get_dataloader():
    """Returns a ramdom dataloader containing 10 batches of 3x5 tensors"""
    return [torch.randn(3, 5) for _ in range(10)]


def get_verify_function():
    """Define verify function that compares outputs of the torch model and the optimized model."""

    def verify_func(ys_runner: Iterable[Sample], ys_expected: Iterable[Sample]) -> bool:
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.allclose(a, b, rtol=1.0e-3, atol=1.0e-3) for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def get_optimization_profile():
    return nav.torch.OptimizationProfile(
        batch_sizes=[1, 2, 4],
        measurement_mode=nav.MeasurementMode.TIME_WINDOWS,
        measurement_interval=2500,  # ms
        measurement_request_count=10,
        stability_percentage=15,
        max_trials=5,
        throughput_cutoff_threshold=0.1,
    )


def get_onnx_config():
    """Get ONNX config for the model export and conversion

    onnx_extended_conversion=True enables additional conversions from TorchScript to ONNX.
    Additional conversion for ONNX means that three different ONNX will be produced:
    - PyTorch -> ONNX
    - TorchScript Script -> ONNX
    - TorchScript Trace -> ONNX

    From each ONNX different TensorRT plan will be produced.
    """
    return nav.OnnxConfig(
        opset=17,
        onnx_extended_conversion=True,
    )


def get_tensortrt_config():
    """Get TensorRT config for the model export and conversion

    Use only FP32 precision and 8589934592 byts (8 GB) of workspace size.
    """
    return nav.TensorRTConfig(
        precision=(nav.TensorRTPrecision.FP32,),
        max_workspace_size=8589934592,
    )


def main():
    """Get model, dataloader, verify_func, and run optimization"""
    model = get_model()
    dataloader = get_dataloader()
    verify_func = get_verify_function()
    optimization_profile = get_optimization_profile()
    onnx_config = get_onnx_config()
    tensortrt_config = get_tensortrt_config()
    """
    Optimize the model by performing model export, conversion, correctness tests,
    profiling and additional verification.

    Convert only to ONNX and TensorRT formats. TorchTensorRT will be omitted.
    Use custom ONNX and TensorRT configurations and custom optimization profile.
    Set custom input names for the model.

    """
    nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        target_formats=(nav.Format.ONNX, nav.Format.TENSORRT),
        input_names=("model_input_0",),
        verify_func=verify_func,  # verify_func is optional but recommended.
        optimization_profile=optimization_profile,
        custom_configs=(
            onnx_config,
            tensortrt_config,
        ),
    )


if __name__ == "__main__":
    main()
