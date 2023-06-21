# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
from copy import copy
from unittest.mock import Mock

import numpy as np
import pytest

from model_navigator.exceptions import (
    ModelNavigatorEmptyPackageError,
    ModelNavigatorError,
    ModelNavigatorRuntimeAnalyzerError,
    ModelNavigatorWrongParameterError,
)
from model_navigator.runtime_analyzer import RuntimeAnalyzer
from model_navigator.runtime_analyzer.analyzer import RuntimeAnalyzerResult
from model_navigator.triton import model_repository
from model_navigator.triton.model_repository import add_model, add_model_from_package
from model_navigator.triton.specialized_configs import (
    DeviceKind,
    InputTensorSpec,
    ONNXModelConfig,
    OutputTensorSpec,
    PythonModelConfig,
    PyTorchModelConfig,
    TensorFlowModelConfig,
    TensorRTAccelerator,
    TensorRTModelConfig,
)
from tests.unit.base.mocks.packages import (
    custom_runner_package,
    empty_package,
    onnx_package_with_cpu_runner_only,
    onnx_package_with_cuda_runner,
    onnx_package_with_tensorrt_runner,
    tensorflow_package_with_tensorflow_only,
    tensorflow_package_with_tensorflow_tensorrt,
    tensorrt_package,
    torchscript_package_with_cpu_only,
    torchscript_package_with_cuda,
    torchscript_package_with_torch_tensorrt,
)
from tests.utils import get_assets_path


def test_add_model_raise_error_when_unsupported_config_passed():
    model_path = get_assets_path() / "models" / "identity.onnx"
    with pytest.raises(ModelNavigatorWrongParameterError, match="Unsupported model config provided: <class 'object'>"):
        add_model(
            model_repository_path=pathlib.Path.cwd(),
            model_name="TestModel",
            model_version=1,
            model_path=model_path,
            config=object(),  # pytype: disable=wrong-arg-types
        )


def test_add_model_raise_error_when_model_not_exists():
    model_path = get_assets_path() / "models" / "identity.notexisting"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        with pytest.raises(OSError):
            add_model(
                model_repository_path=model_repository_path,
                model_name="TestModel",
                model_version=1,
                model_path=model_path,
                config=ONNXModelConfig(),
            )


def test_add_model_create_catalog_in_repository_when_onnx_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.touch()

        add_model(
            model_repository_path=model_repository_path,
            model_name="TestModel",
            model_version=1,
            model_path=model_path,
            config=ONNXModelConfig(),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.onnx").exists()


def test_add_model_create_catalog_in_repository_when_python_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.mkdir()

        python_config = PythonModelConfig(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(-1,)),
                InputTensorSpec(name="INPUT_2", dtype=np.dtype("bytes"), shape=(100, 100)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        )

        add_model(
            model_repository_path=model_repository_path,
            model_name="TestModel",
            model_version=1,
            model_path=model_path,
            config=python_config,
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.py").exists()


def test_add_model_create_catalog_in_repository_when_pytorch_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.mkdir()

        pytorch_config = PyTorchModelConfig(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(-1,)),
                InputTensorSpec(name="INPUT_2", dtype=np.dtype("bytes"), shape=(100, 100)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        )

        add_model(
            model_repository_path=model_repository_path,
            model_name="TestModel",
            model_version=1,
            model_path=model_path,
            config=pytorch_config,
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.pt").exists()


def test_add_model_create_catalog_in_repository_when_tensorflow_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.mkdir()

        add_model(
            model_repository_path=model_repository_path,
            model_name="TestModel",
            model_version=1,
            model_path=model_path,
            config=TensorFlowModelConfig(),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.savedmodel").exists()


def test_add_model_create_catalog_in_repository_when_tensorrt_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.touch()

        add_model(
            model_repository_path=model_repository_path,
            model_name="TestModel",
            model_version=1,
            model_path=model_path,
            config=TensorRTModelConfig(),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.plan").exists()


def test_add_model_create_catalog_in_repository_when_string_path_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.touch()

        add_model(
            model_repository_path=model_repository_path.as_posix(),
            model_name="TestModel",
            model_version=1,
            model_path=model_path.as_posix(),
            config=TensorRTModelConfig(),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.plan").exists()


def test_add_model_from_package_raise_error_when_unsupported_triton_runner_in_package():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        model_name = "Model"
        model_version = 2

        package = custom_runner_package(workspace)
        with pytest.raises(
            ModelNavigatorRuntimeAnalyzerError,
            match="No runtime has both the minimal latency and the maximal throughput."
            "Consider using different `RuntimeSearchStrategy`",
        ):
            add_model_from_package(
                model_repository_path=model_repository_path,
                model_name=model_name,
                model_version=model_version,
                package=package,
            )


def test_add_model_from_package_raises_error_when_package_is_empty():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = empty_package(workspace_path)

        with pytest.raises(ModelNavigatorEmptyPackageError):
            add_model_from_package(
                model_repository_path,
                model_name=model_name,
                model_version=model_version,
                package=package,
            )


def test_add_model_from_package_raises_error_when_unsupported_format_obtained(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = onnx_package_with_tensorrt_runner(workspace_path)
        mocked_model_status = copy(package.status.models_status["onnx"])
        mocked_model_status.model_config = Mock()
        mocked_model_status.model_config.format = object()

        mocked_runner_status = mocked_model_status.runners_status["OnnxCUDA"]

        mocked_result = RuntimeAnalyzerResult(
            latency=1,
            throughput=10,
            model_status=mocked_model_status,
            runner_status=mocked_runner_status,
        )

        with mocker.patch.object(RuntimeAnalyzer, "get_runtime", return_value=mocked_result):
            with pytest.raises(ModelNavigatorError, match="Unsupported model format selected:"):
                add_model_from_package(
                    model_repository_path,
                    model_name=model_name,
                    model_version=model_version,
                    package=package,
                )


def test_add_model_from_package_raises_error_when_batching_dimension_is_invalid():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = onnx_package_with_tensorrt_runner(workspace_path)
        package.status.config["batch_dim"] = 1

        with pytest.raises(
            ModelNavigatorWrongParameterError,
            match="Only models without batching or batch dimension on first place in shape are supported for Triton.",
        ):
            add_model_from_package(
                model_repository_path,
                model_name=model_name,
                model_version=model_version,
                package=package,
            )


def test_add_model_from_package_select_onnx_when_tensorrt_runner_selected(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = onnx_package_with_tensorrt_runner(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, ONNXModelConfig) is True

        assert config.optimization is not None
        assert isinstance(config.optimization.accelerator, TensorRTAccelerator)

        assert config.instance_groups is not None
        assert len(config.instance_groups) == 1
        assert config.instance_groups[0].kind == DeviceKind.KIND_GPU

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.onnx").exists()


def test_add_model_from_package_select_onnx_when_cuda_runner_selected(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = onnx_package_with_cuda_runner(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, ONNXModelConfig) is True

        assert config.optimization is None

        assert config.instance_groups is not None
        assert len(config.instance_groups) == 1
        assert config.instance_groups[0].kind == DeviceKind.KIND_GPU

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.onnx").exists()


def test_add_model_from_package_select_onnx_when_cpu_runner_selected(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = onnx_package_with_cpu_runner_only(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, ONNXModelConfig) is True

        assert config.optimization is None
        assert len(config.instance_groups) == 0

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.onnx").exists()


def test_add_model_from_package_select_tensorflow_when_tensorflow_saved_model_runner_selected(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = tensorflow_package_with_tensorflow_only(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, TensorFlowModelConfig) is True

        assert len(config.instance_groups) == 0

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.savedmodel").exists()


def test_add_model_from_package_select_tensorflow_when_tensorflow_tensorrt_runner_selected(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = tensorflow_package_with_tensorflow_tensorrt(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, TensorFlowModelConfig) is True

        assert len(config.instance_groups) == 1
        assert config.instance_groups[0].kind == DeviceKind.KIND_GPU

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.savedmodel").exists()


def test_add_model_from_package_select_torch_tensorrt_when_torch_tensorrt_model(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = torchscript_package_with_torch_tensorrt(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, PyTorchModelConfig) is True

        assert config.instance_groups is not None
        assert len(config.instance_groups) == 1
        assert config.instance_groups[0].kind == DeviceKind.KIND_GPU

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.pt").exists()


def test_add_model_from_package_select_torchscript_when_cuda_runner_selected(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = torchscript_package_with_cuda(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, PyTorchModelConfig) is True

        assert config.instance_groups is not None
        assert len(config.instance_groups) == 1
        assert config.instance_groups[0].kind == DeviceKind.KIND_GPU

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.pt").exists()


def test_add_model_from_package_select_torchscript_when_cpu_runner_selected(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = torchscript_package_with_cpu_only(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, PyTorchModelConfig) is True

        assert len(config.instance_groups) == 0

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.pt").exists()


def test_add_model_from_package_select_tensorrt_when_tensorrt_package_provided(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = tensorrt_package(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path,
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, TensorRTModelConfig) is True

        assert len(config.instance_groups) == 1
        assert config.instance_groups[0].kind == DeviceKind.KIND_GPU

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.plan").exists()


def test_add_model_from_package_create_model_when_string_path_provided(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package = tensorrt_package(workspace_path)

        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path=model_repository_path.as_posix(),
            model_name=model_name,
            model_version=model_version,
            package=package,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, TensorRTModelConfig) is True

        assert len(config.instance_groups) == 1
        assert config.instance_groups[0].kind == DeviceKind.KIND_GPU

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.plan").exists()
