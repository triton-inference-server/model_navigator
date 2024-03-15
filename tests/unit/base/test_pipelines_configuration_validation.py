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
"""Tests for ConvertONNX2TRT conversion.

Note:
     Those test do not execute the conversion.
     The tests are checking if correct paths are executed on input arguments.
"""

import pathlib
import tempfile

import pytest

from model_navigator.api.config import (
    Format,
    OnnxConfig,
    TensorRTConfig,
    TensorRTProfile,
    TorchConfig,
    TorchTensorRTConfig,
)
from model_navigator.exceptions import ModelNavigatorConfigurationError, ModelNavigatorConfigurationWarning
from model_navigator.frameworks import is_trt_available
from model_navigator.pipelines.validation import PipelineManagerConfigurationValidator
from tests.unit.base.mocks.packages import onnx_package, onnx_package_with_cpu_runner_only


def test_validator_raises_no_errors_when_configuration_is_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        PipelineManagerConfigurationValidator.run(config, None)
        PipelineManagerConfigurationValidator.run(config, package)


def test_validator_raises_error_when_wrong_type_in_configuration():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config.framework = 0
        with pytest.raises(ModelNavigatorConfigurationError):
            PipelineManagerConfigurationValidator.run(config, None)


def test_validator_warns_when_custom_config_format_is_not_in_target_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config.custom_configs = {"torch": TorchConfig()}
        with pytest.warns(ModelNavigatorConfigurationWarning):
            PipelineManagerConfigurationValidator.run(config, None)


def test_validator_raises_error_when_target_formats_does_not_match_farmework():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config.target_formats = (Format.JAX,)
        with pytest.raises(ModelNavigatorConfigurationError):
            PipelineManagerConfigurationValidator.run(config, None)


def test_validator_raises_error_when_batching_is_disabled_and_profiler_specify_batch_sizes():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config.batch_dim = None
        config.optimization_profile.batch_sizes = [1]
        with pytest.raises(ModelNavigatorConfigurationError):
            PipelineManagerConfigurationValidator.run(config, None)


def test_validator_raises_no_error_when_trt_profile_names_match_input_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config._input_names = ("my_input",)
        config.target_formats = (Format.TENSORRT,)
        config.custom_configs = {
            "TensorRT": TensorRTConfig(trt_profiles=[TensorRTProfile().add("my_input", (1,), (2,), (4,))])
        }
        PipelineManagerConfigurationValidator.run(config, None)


def test_validator_raises_error_when_trt_profile_names_mismatch_input_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config._input_names = ("my_input",)
        config.target_formats = (Format.TORCH_TRT,)
        config.custom_configs = {
            "TorchTRT": TorchTensorRTConfig(trt_profiles=[TensorRTProfile().add("not_my_input", (1,), (2,), (4,))])
        }
        with pytest.raises(ModelNavigatorConfigurationError):
            PipelineManagerConfigurationValidator.run(config, None)


def test_validator_raises_no_error_when_trt_profile_batch_dimension_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config.target_formats = (Format.TENSORRT,)
        config.custom_configs = {
            "TensorRT": TensorRTConfig(
                trt_profiles=[TensorRTProfile().add("my_input", (1,), (2,), (4,)).add("my_input_2", (1,), (2,), (4,))]
            )
        }
        PipelineManagerConfigurationValidator.run(config, None)


def test_validator_raises_error_when_trt_profile_batch_dimension_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        workspace = tmpdir / "navigator_workspace"
        package = onnx_package_with_cpu_runner_only(workspace)
        config = package.config
        config.target_formats = (Format.TORCH_TRT,)
        config.custom_configs = {
            "TorchTRT": TorchTensorRTConfig(
                trt_profiles=[TensorRTProfile().add("my_input", (1,), (2,), (4,)).add("my_input_2", (1,), (2,), (8,))]
            )
        }
        with pytest.raises(ModelNavigatorConfigurationError):
            PipelineManagerConfigurationValidator.run(config, None)


def test_validator_raises_no_error_when_target_format_source_is_saved_in_pacakge():
    if is_trt_available():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            workspace = tmpdir / "navigator_workspace"
            package = onnx_package(workspace)
            config = package.config
            config.target_formats = (Format.ONNX,)
            PipelineManagerConfigurationValidator.run(config, package)


def test_validator_warns_when_target_format_source_is_not_saved_in_pacakge():
    if is_trt_available():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            workspace = tmpdir / "navigator_workspace"
            package = onnx_package_with_cpu_runner_only(workspace)
            config = package.config
            config.target_formats = (Format.TENSORRT,)
            with pytest.warns(ModelNavigatorConfigurationWarning):
                PipelineManagerConfigurationValidator.run(config, package)


def test_validator_raises_no_error_when_trt_profile_aligns_with_dynamic_axes():
    if is_trt_available():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            workspace = tmpdir / "navigator_workspace"
            package = onnx_package(workspace)
            config = package.config
            config.custom_configs["Onnx"] = OnnxConfig(dynamic_axes={"my_input": [0, 1]})
            config.custom_configs["TensorRT"] = TensorRTConfig(
                trt_profiles=[TensorRTProfile().add("my_input", (1, 1), (2, 2), (4, 4))]
            )
            PipelineManagerConfigurationValidator.run(config, package)


def test_validator_raises_error_when_trt_profile_does_not_align_with_dynamic_axes():
    if is_trt_available():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            workspace = tmpdir / "navigator_workspace"
            package = onnx_package(workspace)
            config = package.config
            config.custom_configs["Onnx"] = OnnxConfig(dynamic_axes={"my_input": [0]})
            config.custom_configs["TensorRT"] = TensorRTConfig(
                trt_profiles=[TensorRTProfile().add("my_input", (1, 1), (2, 2), (4, 4))]
            )
            with pytest.raises(ModelNavigatorConfigurationError):
                PipelineManagerConfigurationValidator.run(config, package)
