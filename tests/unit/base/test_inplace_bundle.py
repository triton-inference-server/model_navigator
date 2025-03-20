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
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from model_navigator.configuration import Format
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.inplace import bundle
from model_navigator.inplace.registry import ModuleRegistry
from tests.unit.base.mocks import packages as mock_packages


@pytest.fixture
def nav_cache(mocker, tmp_path: Path):
    nav_cache = tmp_path / ".cache"
    nav_cache.mkdir(exist_ok=True)

    mocker.patch("model_navigator.inplace.bundle.inplace_cache_dir", lambda: nav_cache)

    return nav_cache


def create_mock_module(nav_cache: Path, name: str, is_optimized: bool, status_yaml: str):
    m1 = MagicMock()
    m1.is_optimized = is_optimized
    m1_path = nav_cache / name

    m1_path.mkdir(parents=True, exist_ok=True)
    (m1_path / "status.yaml").write_text(status_yaml)

    return m1


@pytest.mark.parametrize(
    "select_modules",
    [
        bundle.AllModulesSelection(),  # modules selection strategies
        bundle.RegisteredModulesSelection(),
        bundle.ModulesByNameSelection(["m1"]),
    ],
)
def test_bundle_save(mocker, nav_cache: Path, select_modules):
    # register mock modules
    m1 = create_mock_module(nav_cache, "m1", is_optimized=True, status_yaml="optimized: true")
    m2 = create_mock_module(nav_cache, "m2", is_optimized=True, status_yaml="optimized: false")
    _m3 = create_mock_module(nav_cache, "m3", is_optimized=True, status_yaml="optimized: true")

    # create mock registry
    module_registry = ModuleRegistry()
    module_registry.register("m1", m1)
    module_registry.register("m2", m2)

    bundle_file_result = nav_cache / "bundle.nav"

    mocker.patch("model_navigator.inplace.bundle.module_registry", module_registry)

    bundle.save(bundle_file_result, modules=select_modules)

    assert bundle_file_result.exists()

    with zipfile.ZipFile(bundle_file_result, "r") as zip_file:
        assert "m1/status.yaml" in zip_file.namelist(), f"status.yml for m1 not found in zip; {zip_file.namelist()}"
        assert zip_file.read("m1/status.yaml").decode() == "optimized: true"

        if isinstance(select_modules, bundle.ModulesByNameSelection):
            assert "m2/status.yaml" not in zip_file.namelist(), "This module was not selected by name!"
        else:
            assert "m2/status.yaml" in zip_file.namelist(), f"status.yml for m2 not found in zip; {zip_file.namelist()}"
            assert zip_file.read("m2/status.yaml").decode() == "optimized: false"

        if isinstance(select_modules, bundle.AllModulesSelection):
            assert "m3/status.yaml" in zip_file.namelist(), f"status.yml for m3 not found in zip; {zip_file.namelist()}"
        else:
            assert (
                "m3/status.yaml" not in zip_file.namelist()
            ), f"status.yaml for m3 found in zip but shouldn't; {zip_file.namelist()}"


def test__major_minor_version():
    assert (0, 0) == bundle._major_minor_version("0.0.0")
    assert (1, 0) == bundle._major_minor_version("1")
    assert (2, 1) == bundle._major_minor_version("2.1")
    assert (2, 1) == bundle._major_minor_version("2.1.100")
    assert (2, 1) == bundle._major_minor_version("2.1.100+dev")


@pytest.fixture
def bundle_path_m1(nav_cache: Path, status_yaml, tmp_path):
    m1 = MagicMock()
    m1.is_optimized = True
    m1_path = nav_cache / "m1"
    m1_status = m1_path / "status.yaml"

    m1_path.mkdir(parents=True, exist_ok=True)
    m1_status.write_text(status_yaml)

    bundle_path = nav_cache / "bundle.nav"

    bundle.save(bundle_path, modules=bundle.AllModulesSelection(), tags=["tag1", "tag2"])

    return bundle_path


@pytest.fixture
def status_yaml():
    return """format_version: 0.3.1
model_navigator_version: 0.8.1
environment:
  cpu:
    name: 13th Gen Intel(R) Core(TM) i9-13900K
    physical_cores: 24
    logical_cores: 32
    min_frequency: 800.0
    max_frequency: 4937.5
  memory: 62.5G
  gpu:
    name: NVIDIA RTX A6000
    driver_version: 545.29.06
    memory: 49140 MiB
    tdp: 300.00 W
    cuda_version: '12.3'
  os:
    name: posix
    platform: Linux
    release: 6.5.0-17-generic
  python_version: 3.10.12
  python_packages:
    numpy: 1.24.4
    onnx: 1.16.0
    onnxruntime-gpu: 1.17.1
    polygraphy: 0.49.7
    tensorboard: 2.9.0
    tensorboard-data-server: 0.6.1
    tensorboard-plugin-wit: 1.8.1
    tensorrt: 8.6.3
    torch: 2.3.0a0+40ec155e58.nv24.3
    torch-tensorrt: 2.3.0a0
    torchtext: 0.17.0a0
    torchvision: 0.18.0a0
    tritonclient: 2.44.0
    xgboost: 2.0.3
  libraries:
    NPP_VERSION: 12.2.5.2
    DALI_BUILD: '12768324'
    CUSOLVER_VERSION: 11.6.0.99
    CUBLAS_VERSION: 12.4.2.65
    CUFFT_VERSION: 11.2.0.44
    NCCL_VERSION: 2.20.5
    CUSPARSE_VERSION: 12.3.0.142
    OPENUCX_VERSION: 1.16.0
    NSIGHT_SYSTEMS_VERSION: 2024.2.1.38
    TRT_VERSION: 8.6.3.1+cuda12.2.2.009
    CUDA_VERSION: 12.4.0.041
    PYTORCH_VERSION: 2.3.0a0+40ec155e58
    CURAND_VERSION: 10.3.5.119
    OPENMPI_VERSION: 4.1.7
    NVJPEG_VERSION: 12.3.1.89
    CUDNN_VERSION: 9.0.0.306+cuda12.3
    NSIGHT_COMPUTE_VERSION: 2024.1.0.13
    DALI_VERSION: 1.35.0
    NVIDIA_BUILD_ID: '85286408'
    CUDA_DRIVER_VERSION: 550.54.14
    NVIDIA_PYTORCH_VERSION: '24.03'
    TRTOSS_VERSION: '23.11'
config:
  framework: torch
  target_formats:
  - torch
  - trt
  target_device: cuda
  sample_count: 100
  optimization_profile:
    max_batch_size: 64
    batch_sizes: null
    window_size: 50
    stability_percentage: 10.0
    stabilization_windows: 3
    min_trials: 3
    max_trials: 10
    throughput_cutoff_threshold: 0.05
  runner_names:
  - TensorRT
  batch_dim: 0
  seed: 0
  _input_names: null
  _output_names: null
  from_source: true
  custom_configs:
    Torch:
      custom_args: {}
      device: null
      autocast: true
      inference_mode: true
    TorchScript:
      custom_args: {}
      device: null
      jit_type:
      - script
      - trace
      strict: true
      autocast: true
      inference_mode: true
    TensorRT:
      custom_args: {}
      device: null
      trt_profiles:
      - input__0:
          min:
          - 1
          - 24
          opt:
          - 8
          - 304
          max:
          - 64
          - 512
        input__1:
          min:
          - 1
          - 24
          opt:
          - 8
          - 304
          max:
          - 64
          - 512
      trt_profile: null
      precision:
      - fp16
      precision_mode: hierarchy
      max_workspace_size: 8589934592
      run_max_batch_size_search: true
      optimization_level: null
      compatibility_level: null
      onnx_parser_flags:
      - 0
    Onnx:
      custom_args: {}
      device: null
      opset: 17
      dynamo_export: false
      dynamic_axes:
        input__0:
        - 0
        - 1
        input__1:
        - 0
        - 1
      onnx_extended_conversion: false
      graph_surgeon_optimization: true
      export_device: null
    TorchTensorRT:
      custom_args: {}
      device: null
      trt_profiles:
      - input__0:
          min:
          - 1
          - 24
          opt:
          - 8
          - 304
          max:
          - 64
          - 512
        input__1:
          min:
          - 1
          - 24
          opt:
          - 8
          - 304
          max:
          - 64
          - 512
      trt_profile: null
      precision:
      - fp32
      - fp16
      precision_mode: hierarchy
      max_workspace_size: 8589934592
      run_max_batch_size_search: true
  verbose: false
  debug: false
module_status:
  e5-large-full.0:
    uuid: 41573c38-16b4-11ef-97b5-cc96e53ddb09
    models_status:
      torchscript-script:
        model_config:
          format: torchscript
      torchscript-trace:
        model_config:
          format: torchscript
      onnx:
        model_config:
          format: onnx
      trt-fp16:
        model_config:
          format: trt
"""


@pytest.fixture
def nav_env(mocker, status_yaml):
    env_vars = yaml.safe_load(status_yaml)["environment"]

    mocker.patch("model_navigator.inplace.bundle.get_env", lambda: env_vars)

    return env_vars


def test_bundle_match_check(bundle_path_m1, nav_env):
    assert bundle.is_matching(bundle_path_m1)


def test_bundle_match_check_with_tags(bundle_path_m1, nav_env):
    assert bundle.is_matching(bundle_path_m1, tags=["tag1", "tag2"])


def test_bundle_match_check_tags_mismatch(bundle_path_m1, nav_env):
    assert not bundle.is_matching(bundle_path_m1, tags=["tag1"])


@pytest.fixture
def simple_bundle_path(tmp_path: Path, status_yaml):
    bundle_file = tmp_path / "bundle.nav"
    with zipfile.ZipFile(bundle_file, "w") as zip_file:
        zip_file.writestr("tags.yaml", "tags: [tag1, tag2]")
        zip_file.writestr("m1/0/status.yaml", status_yaml)
        zip_file.writestr("m2/0/status.yaml", status_yaml)
        zip_file.writestr("m3/0/status.yaml", status_yaml)

    return bundle_file


def test_load_cache_bundle(nav_cache: Path, simple_bundle_path, nav_env):
    bundle.load(simple_bundle_path)

    assert (nav_cache / "m1/0/status.yaml").exists()
    assert (nav_cache / "m2/0/status.yaml").exists()
    assert (nav_cache / "m3/0/status.yaml").exists()
    assert (nav_cache / "tags.yaml").exists()


def test_load_cache_bundle_with_tags(nav_cache: Path, simple_bundle_path, nav_env):
    bundle.load(simple_bundle_path, tags=["tag1", "tag2"])

    assert (nav_cache / "m1/0/status.yaml").exists()
    assert (nav_cache / "m2/0/status.yaml").exists()
    assert (nav_cache / "m3/0/status.yaml").exists()
    assert (nav_cache / "tags.yaml").exists()


def test_load_cache_bundle_tags_mismatch(nav_cache: Path, simple_bundle_path, nav_env):
    with pytest.raises(ModelNavigatorConfigurationError):
        bundle.load(simple_bundle_path, tags=["tag1", "tag4"])


def test_load_cache_bundle_env_mismatch(nav_cache: Path, simple_bundle_path, nav_env):
    nav_env["gpu"]["name"] = "RTX 4090"

    with pytest.raises(ModelNavigatorConfigurationError):
        bundle.load(simple_bundle_path)


def test_load_cache_bundle_remove_prev_module_dir(nav_cache: Path, simple_bundle_path, nav_env):
    # this file should disappear
    old_module = nav_cache / "m1/0/old_file.yaml"
    old_module.parent.mkdir(parents=True, exist_ok=True)
    old_module.write_text("old values")

    bundle.load(simple_bundle_path)

    assert (nav_cache / "m1/0/status.yaml").exists()
    assert not old_module.exists()


@pytest.fixture
def module_with_onnx(mocker, nav_cache: Path, status_yaml):
    workspace_path = nav_cache / "m1/0"
    workspace_path.mkdir(parents=True, exist_ok=True)

    # this files should be included in the bundle
    (workspace_path / "status.yaml").write_text(status_yaml)
    (workspace_path / "context.yaml").write_text("metadata: ok")
    (workspace_path / "navigator.log").write_text("ok")
    (workspace_path / "model_input").mkdir()
    (workspace_path / "model_input/test.log").write_text("ok")
    (workspace_path / "model_output").mkdir()
    (workspace_path / "model_output/test.log").write_text("ok")

    # using mock packages
    m1_package = mock_packages.onnx_package(workspace_path)

    # (sanity check) we need at least 2 runners
    assert "onnx" in m1_package.status.models_status
    assert "trt-fp16" in m1_package.status.models_status, "trt-fp16 should be present in the onnx_package() mock!"

    # mock module wrapper
    wrapper = MagicMock()
    wrapper._packages = [m1_package]

    # mock module
    m1 = MagicMock()
    m1.is_optimized = True
    m1._wrapper = wrapper

    # mock registry
    module_registry = ModuleRegistry()
    module_registry.register("m1", m1)

    mocker.patch("model_navigator.inplace.bundle.module_registry", module_registry)

    return m1, m1_package


def test_save_best_packages(module_with_onnx, tmp_path):
    selection = bundle.BestRunnersSelection()

    _, m1_package = module_with_onnx

    bundle_file_result = tmp_path / "bundle.nav"

    # (sanity check) we need at least 2 runners
    assert "onnx" in m1_package.status.models_status
    assert "trt-fp16" in m1_package.status.models_status, "trt-fp16 should be present in the onnx_package() mock!"

    # (sanity check) best runner is ONNX
    runtime_result = m1_package.get_best_runtime(strategies=selection.runner_selection_strategies, inplace=True)
    assert runtime_result.model_status.model_config.format == Format.ONNX

    bundle.save(bundle_file_result, modules=bundle.BestRunnersSelection())

    assert bundle_file_result.exists()

    with zipfile.ZipFile(bundle_file_result, "r") as zip_file:
        files = set(zip_file.namelist())

        assert "m1/0/onnx/model.onnx" in files, f"ONNX runner should be included in the bundle {files}"
        assert "m1/0/status.yaml" in files
        assert "m1/0/context.yaml" in files

        assert "m1/0/trt-fp16/model.plan" not in files


def test_save_bundle_with_exclude_patterns(module_with_onnx, tmp_path: Path):
    bundle_file_result = tmp_path / "bundle.nav"
    bundle.save(bundle_file_result, modules=bundle.AllModulesSelection(), exclude_patterns=[".*model.plan"])

    with zipfile.ZipFile(bundle_file_result, "r") as zip_file:
        assert "m1/0/onnx/model.onnx" in zip_file.namelist()

        assert "m1/0/trt-fp16/model.plan" not in zip_file.namelist()
        assert "m1/0/status.yaml" in zip_file.namelist()
        assert "m1/0/context.yaml" in zip_file.namelist()
        assert "m1/0/navigator.log" in zip_file.namelist()
        assert "m1/0/model_input/test.log" in zip_file.namelist()
        assert "m1/0/model_output/test.log" in zip_file.namelist()


def test_save_bundle_with_include_patterns(module_with_onnx, tmp_path: Path):
    bundle_file_result = tmp_path / "bundle.nav"
    bundle.save(bundle_file_result, modules=bundle.AllModulesSelection(), include_patterns=[".*model.plan"])

    with zipfile.ZipFile(bundle_file_result, "r") as zip_file:
        assert "m1/0/trt-fp16/model.plan" in zip_file.namelist()

        assert "m1/0/onnx/model.onnx" not in zip_file.namelist()
        assert "m1/0/status.yaml" not in zip_file.namelist()
        assert "m1/0/context.yaml" not in zip_file.namelist()
        assert "m1/0/navigator.log" not in zip_file.namelist()
        assert "m1/0/model_input/test.log" not in zip_file.namelist()
        assert "m1/0/model_output/test.log" not in zip_file.namelist()
