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

import itertools
import pathlib
import tempfile

import pytest

from model_navigator.api.config import Format, JitType, OptimizationProfile, TorchConfig
from model_navigator.api.package import _get_model_configs, _update_config, optimize
from model_navigator.exceptions import ModelNavigatorEmptyPackageError, ModelNavigatorMissingSourceModelError
from model_navigator.runners.registry import runner_registry
from model_navigator.utils import enums
from tests.unit.base.mocks.packages import (
    empty_package,
    onnx_package,
    torchscript_package_with_torch_tensorrt,
    trochscript_package_with_source,
)


def test_get_model_configs_returns_original_configs_when_no_custom_configs_passed():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = torchscript_package_with_torch_tensorrt(workspace)
        model_configs = _get_model_configs(config=package.config, custom_configs=[])
        flatten_model_configs = list(itertools.chain(*list(model_configs.values())))

        assert len(flatten_model_configs) == 6


def test_get_model_configs_returns_updated_torchscript_config_when_torch_custom_config_passed():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = trochscript_package_with_source(workspace)
        model_configs = _get_model_configs(
            config=package.config,
            custom_configs=[
                TorchConfig(jit_type=(JitType.TRACE,)),
            ],
        )

        flatten_model_configs = list(itertools.chain(*list(model_configs.values())))

        assert len(flatten_model_configs) == 2
        for model_config in flatten_model_configs:
            if model_config.key == "torchscript-trace":
                assert model_config.jit_type == JitType.TRACE  # pytype: disable=attribute-error


def test_update_config_returns_updated_custom_config_when_defaults_is_true():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = trochscript_package_with_source(workspace)
        config = package.config
        _update_config(
            config=config,
            is_source_available=True,
            target_formats=(Format.TORCHSCRIPT,),
        )

        assert len(config.custom_configs) == 3
        onnx_config = config.custom_configs["Onnx"]
        torch_config = config.custom_configs["Torch"]
        tensorrt_config = config.custom_configs["TensorRT"]

        assert onnx_config.opset == 13  # pytype: disable=attribute-error
        assert torch_config.jit_type == (JitType.SCRIPT, JitType.TRACE)  # pytype: disable=attribute-error
        assert tensorrt_config.trt_profiles is None  # pytype: disable=attribute-error


def test_update_config_returns_original_config_when_no_parameters_passed_and_source_available():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = torchscript_package_with_torch_tensorrt(workspace)
        config = package.config
        _update_config(
            config,
            is_source_available=True,
            target_formats=(
                Format.TORCH,
                Format.TORCHSCRIPT,
            ),
            defaults=False,
        )

        reset_fields = [
            "target_device",
            "target_formats",
            "runner_names",
            "optimization_profile",
            "from_source",
            "timestamp",
        ]

        runners = tuple(runner_registry.keys())
        runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

        assert config.target_formats == (Format.TORCH, Format.TORCHSCRIPT)
        assert config.runner_names == runner_names
        assert config.optimization_profile == OptimizationProfile()
        assert config.from_source is False

        for key, value in config.to_dict().items():
            if key in reset_fields:
                continue

            assert getattr(package.config, key) == value


def test_update_config_returns_config_wo_source_when_serialized_target_format_passed_and_source_not_available():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = trochscript_package_with_source(workspace)
        config = package.config
        _update_config(config, is_source_available=False, target_formats=(Format.TORCHSCRIPT,), defaults=False)

        reset_fields = [
            "target_device",
            "target_formats",
            "runner_names",
            "optimization_profile",
            "from_source",
            "timestamp",
        ]

        runners = tuple(runner_registry.keys())
        runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

        assert config.target_formats == (Format.TORCHSCRIPT,)
        assert config.runner_names == runner_names
        assert config.optimization_profile == OptimizationProfile()
        assert config.from_source is False

        for key, value in config.to_dict().items():
            if key in reset_fields:
                continue

            assert getattr(package.config, key) == value


def test_update_config_returns_updated_config_when_parameters_passed():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = torchscript_package_with_torch_tensorrt(workspace)
        config = package.config
        kwargs = {
            "runners": ("CustomRunner1", "CustomRunner2"),
            "optimization_profile": "config",
            "verify_func": "verify_func",
        }
        _update_config(
            config,
            is_source_available=False,
            target_formats=(Format.TORCH_TRT,),
            **kwargs,
        )  # pytype: disable=wrong-arg-types

        for key in kwargs:
            config_key = "runner_names" if key == "runners" else key
            assert getattr(config, config_key) == kwargs[key]


def test_update_config_raises_missing_source_error_when_source_format_passed_and_source_not_available():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = torchscript_package_with_torch_tensorrt(workspace)
        config = package.config
        with pytest.raises(ModelNavigatorMissingSourceModelError):
            _update_config(
                config,
                is_source_available=False,
                target_formats=(Format.TORCH,),
            )


def test_update_config_update_config_when_onnx_target_format_passed_and_onnx_file_is_available():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = onnx_package(workspace)
        config = package.config

        _update_config(
            config,
            is_source_available=False,
            target_formats=(Format.ONNX,),
        )

        assert config.target_formats == (Format.ONNX,)


def test_optimize_raises_empty_package_error_when_package_is_empty_and_source_not_available():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        package = empty_package(workspace)
        with pytest.raises(ModelNavigatorEmptyPackageError):
            optimize(package)
