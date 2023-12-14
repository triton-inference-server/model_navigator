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

from model_navigator.core.tensor import TensorMetadata
from model_navigator.exceptions import (
    ModelNavigatorEmptyPackageError,
    ModelNavigatorError,
    ModelNavigatorRuntimeAnalyzerError,
    ModelNavigatorWrongParameterError,
)
from model_navigator.runtime_analyzer import RuntimeAnalyzer
from model_navigator.runtime_analyzer.analyzer import RuntimeAnalyzerResult
from model_navigator.triton import model_repository
from model_navigator.triton.model_config import ModelConfig
from model_navigator.triton.model_repository import (
    _input_tensor_from_metadata,
    _output_tensor_from_metadata,
    _TritonModelRepository,
    add_model,
    add_model_from_package,
)
from model_navigator.triton.specialized_configs import (
    DeviceKind,
    InputTensorSpec,
    ModelWarmup,
    ModelWarmupInput,
    ModelWarmupInputDataType,
    ONNXModelConfig,
    OutputTensorSpec,
    PythonModelConfig,
    PyTorchModelConfig,
    SequenceBatcher,
    SequenceBatcherInitialState,
    SequenceBatcherState,
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


def test_add_model_raise_error_when_model_not_exists_and_backand_support_catalog_based_models():
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
                config=TensorFlowModelConfig(),
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


def test_add_model_create_catalog_in_repository_when_onnx_model_passed_and_default_filename_provided():
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
            config=ONNXModelConfig(default_model_filename="mymodel.onnx"),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "mymodel.onnx").exists()


def test_add_model_create_catalog_in_repository_when_python_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model.py"
        model_path.touch()

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


def test_add_model_create_catalog_in_repository_when_python_model_passed_and_default_filename_provided():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model.py"
        model_path.touch()

        python_config = PythonModelConfig(
            default_model_filename="mymodel.py",
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
        assert (model_repository_path / "TestModel" / "1" / "mymodel.py").exists()


def test_add_model_create_catalog_in_repository_when_catalog_with_python_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.mkdir()

        file1 = model_path / "model.py"
        file1.touch()

        file2 = model_path / "data.py"
        file2.touch()

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
        assert (model_repository_path / "TestModel" / "1" / "data.py").exists()


def test_add_model_create_catalog_in_repository_when_pytorch_model_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model.pt"
        model_path.touch()

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


def test_add_model_create_catalog_in_repository_when_pytorch_model_passed_and_default_model_filename_provided():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model.pt"
        model_path.touch()

        pytorch_config = PyTorchModelConfig(
            default_model_filename="mymodel.pt",
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
        assert (model_repository_path / "TestModel" / "1" / "mymodel.pt").exists()


def test_add_model_create_catalog_in_repository_when_pytorch_model_catalog_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.mkdir()

        file1 = model_path / "file.pt"
        file1.touch()

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
        assert (model_repository_path / "TestModel" / "1" / "model.pt" / "file.pt").exists()


def test_add_model_create_catalog_in_repository_when_pytorch_model_catalog_passed_and_default_model_filename_provided():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.mkdir()

        file1 = model_path / "file.pt"
        file1.touch()

        pytorch_config = PyTorchModelConfig(
            default_model_filename="mymodel.pt",
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
        assert (model_repository_path / "TestModel" / "1" / "mymodel.pt").exists()
        assert (model_repository_path / "TestModel" / "1" / "mymodel.pt" / "file.pt").exists()


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


def test_add_model_create_catalog_in_repository_when_tensorflow_model_passed_and_default_model_filename_provided():
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
            config=TensorFlowModelConfig(
                default_model_filename="mymodel.savemodel",
            ),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "mymodel.savemodel").exists()


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


def test_add_model_create_catalog_in_repository_when_tensorrt_model_passed_and_default_model_filename_provided():
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
            config=TensorRTModelConfig(
                default_model_filename="mymodel.plan",
            ),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "mymodel.plan").exists()


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


def test_add_model_create_model_with_warmup_when_enabled(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        model_path = tmpdir / "model"
        model_path.touch()

        spy_deploy_model = mocker.spy(_TritonModelRepository, "deploy_model")

        add_model(
            model_repository_path=model_repository_path.as_posix(),
            model_name="TestModel",
            model_version=1,
            model_path=model_path.as_posix(),
            config=TensorRTModelConfig(
                warmup={
                    "warmup_1": ModelWarmup(
                        inputs={
                            "input_0": ModelWarmupInput(
                                shape=(1,),
                                dtype=np.dtype("float32"),
                                input_data_type=ModelWarmupInputDataType.ZERO,
                            )
                        }
                    ),
                    "warmup_2": ModelWarmup(
                        batch_size=2,
                        iterations=1,
                        inputs={
                            "input_0": ModelWarmupInput(
                                shape=(1,),
                                dtype=np.dtype("float32"),
                                input_data_type=ModelWarmupInputDataType.RANDOM,
                            )
                        },
                    ),
                }
            ),
        )

        config = spy_deploy_model.call_args.kwargs["model_config"]

        assert isinstance(config, ModelConfig) is True

        assert config.warmup["warmup_1"].batch_size == 1
        assert config.warmup["warmup_1"].iterations == 0
        assert config.warmup["warmup_1"].inputs["input_0"].shape == (1,)
        assert config.warmup["warmup_1"].inputs["input_0"].dtype == np.dtype("float32")
        assert config.warmup["warmup_1"].inputs["input_0"].input_data_type.value == ModelWarmupInputDataType.ZERO.value

        assert config.warmup["warmup_2"].batch_size == 2
        assert config.warmup["warmup_2"].iterations == 1
        assert config.warmup["warmup_2"].inputs["input_0"].shape == (1,)
        assert config.warmup["warmup_2"].inputs["input_0"].dtype == np.dtype("float32")
        assert (
            config.warmup["warmup_2"].inputs["input_0"].input_data_type.value == ModelWarmupInputDataType.RANDOM.value
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.plan").exists()


def test_add_model_create_model_with_warmup_when_enabled_and_warmup_file_passed(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        warmup_file1 = tmpdir / "file1.data"
        warmup_file1.touch()

        warmup_file2 = tmpdir / "file2.data"
        warmup_file2.touch()

        model_path = tmpdir / "model"
        model_path.touch()

        add_model(
            model_repository_path=model_repository_path.as_posix(),
            model_name="TestModel",
            model_version=1,
            model_path=model_path.as_posix(),
            config=TensorRTModelConfig(
                warmup={
                    "warmup_1": ModelWarmup(
                        inputs={
                            "input_0": ModelWarmupInput(
                                shape=(1,),
                                dtype=np.dtype("float32"),
                                input_data_type=ModelWarmupInputDataType.FILE,
                                input_data_file=warmup_file1,
                            )
                        }
                    ),
                    "warmup_2": ModelWarmup(
                        batch_size=2,
                        iterations=1,
                        inputs={
                            "input_0": ModelWarmupInput(
                                shape=(1,),
                                dtype=np.dtype("float32"),
                                input_data_type=ModelWarmupInputDataType.FILE,
                                input_data_file=warmup_file2,
                            )
                        },
                    ),
                }
            ),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.plan").exists()

        assert (model_repository_path / "TestModel" / "warmup").exists()
        assert (model_repository_path / "TestModel" / "warmup" / "file1.data").exists()
        assert (model_repository_path / "TestModel" / "warmup" / "file2.data").exists()


def test_add_model_create_model_when_sequence_batcher_initial_state_file(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        model_repository_path = tmpdir / "model-repository"
        model_repository_path.mkdir()

        initial_state1 = tmpdir / "file1.data"
        initial_state1.touch()

        initial_state2 = tmpdir / "file2.data"
        initial_state2.touch()

        model_path = tmpdir / "model"
        model_path.touch()

        add_model(
            model_repository_path=model_repository_path.as_posix(),
            model_name="TestModel",
            model_version=1,
            model_path=model_path.as_posix(),
            config=TensorRTModelConfig(
                batcher=SequenceBatcher(
                    states=[
                        SequenceBatcherState(
                            input_name="input_1",
                            output_name="output_1",
                            dtype=np.float32,
                            shape=(-1,),
                            initial_states=[
                                SequenceBatcherInitialState(
                                    name="initialization",
                                    dtype=np.int32,
                                    shape=(-1, -1),
                                    data_file=initial_state1,
                                )
                            ],
                        ),
                        SequenceBatcherState(
                            input_name="input_2",
                            output_name="output_2",
                            dtype=np.int32,
                            shape=(-1, -1),
                            initial_states=[
                                SequenceBatcherInitialState(
                                    name="initialization", dtype=np.int32, shape=(-1, -1), data_file=initial_state2
                                )
                            ],
                        ),
                    ]
                ),
            ),
        )

        assert (model_repository_path / "TestModel" / "config.pbtxt").exists()
        assert (model_repository_path / "TestModel" / "1" / "model.plan").exists()

        assert (model_repository_path / "TestModel" / "initial_state").exists()
        assert (model_repository_path / "TestModel" / "initial_state" / "file1.data").exists()
        assert (model_repository_path / "TestModel" / "initial_state" / "file2.data").exists()


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


def test_add_model_from_package_create_model_with_warmup_when_model_has_static_shapes(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = pathlib.Path(tmp_dir) / "workspace"
        model_repository_path = pathlib.Path(tmp_dir) / "model_repository"

        model_name = "Model"
        model_version = 1

        package_path = get_assets_path() / "packages" / "onnx_identity.nav"
        import model_navigator as nav

        package = nav.package.load(package_path, workspace=workspace_path)
        spy_add_model = mocker.spy(model_repository, "add_model")

        add_model_from_package(
            model_repository_path=model_repository_path.as_posix(),
            model_name=model_name,
            model_version=model_version,
            package=package,
            warmup=True,
        )

        config = spy_add_model.call_args.kwargs["config"]

        assert isinstance(config, ONNXModelConfig) is True

        assert config.warmup["warmup_1"].batch_size == 1
        assert config.warmup["warmup_1"].iterations == 0
        assert config.warmup["warmup_1"].inputs["X"].shape == [3, 8, 8]
        assert config.warmup["warmup_1"].inputs["X"].dtype == np.dtype("float32")
        assert config.warmup["warmup_1"].inputs["X"].input_data_type.value == ModelWarmupInputDataType.FILE.value
        assert config.warmup["warmup_1"].inputs["X"].input_data_file == workspace_path / "warmup" / "X.data"

        assert config.warmup["warmup_16"].batch_size == 16
        assert config.warmup["warmup_16"].iterations == 0
        assert config.warmup["warmup_16"].inputs["X"].shape == [3, 8, 8]
        assert config.warmup["warmup_16"].inputs["X"].dtype == np.dtype("float32")
        assert config.warmup["warmup_16"].inputs["X"].input_data_type.value == ModelWarmupInputDataType.FILE.value
        assert config.warmup["warmup_16"].inputs["X"].input_data_file == workspace_path / "warmup" / "X.data"

        assert (model_repository_path / "Model" / "config.pbtxt").exists()
        assert (model_repository_path / "Model" / "1" / "model.onnx").exists()

        assert (model_repository_path / "Model" / "warmup").exists()
        assert (model_repository_path / "Model" / "warmup" / "X.data").exists()


# def test_add_model_from_package_create_model_with_warmup_when_model_has_dynamic_shapes(mocker):
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         workspace_path = pathlib.Path(tmp_dir) / "workspace"
#         model_repository_path = pathlib.Path(tmp_dir) / "model_repository"
#
#         model_name = "Model"
#         model_version = 1
#
#         package_path = get_assets_path() / "packages" / "torch_identity.nav"
#         import model_navigator as nav
#
#         package = nav.package.load(package_path, workspace=workspace_path)
#         spy_add_model = mocker.spy(model_repository, "add_model")
#
#         add_model_from_package(
#             model_repository_path=model_repository_path.as_posix(),
#             model_name=model_name,
#             model_version=model_version,
#             package=package,
#             warmup=True,
#         )
#
#         config = spy_add_model.call_args.kwargs["config"]
#
#         assert isinstance(config, ONNXModelConfig) is True
#
#         assert config.warmup["warmup_1"].batch_size == 1
#         assert config.warmup["warmup_1"].iterations == 0
#         assert config.warmup["warmup_1"].inputs["input_0"].shape == [3]
#         assert config.warmup["warmup_1"].inputs["input_0"].dtype == np.dtype("float32")
#         assert config.warmup["warmup_1"].inputs["input_0"].input_data_type.value == ModelWarmupInputDataType.FILE.value
#         assert config.warmup["warmup_1"].inputs["input_0"].input_data_file == workspace_path / "warmup" / "input_0.data"
#
#         assert config.warmup["warmup_16"].batch_size == 16
#         assert config.warmup["warmup_16"].iterations == 0
#         assert config.warmup["warmup_16"].inputs["input_0"].shape == [3]
#         assert config.warmup["warmup_16"].inputs["input_0"].dtype == np.dtype("float32")
#         assert config.warmup["warmup_16"].inputs["input_0"].input_data_type.value == ModelWarmupInputDataType.FILE.value
#         assert (
#             config.warmup["warmup_16"].inputs["input_0"].input_data_file == workspace_path / "warmup" / "input_0.data"
#         )
#
#         assert (model_repository_path / "Model" / "config.pbtxt").exists()
#         assert (model_repository_path / "Model" / "1" / "model.onnx").exists()
#
#         assert (model_repository_path / "Model" / "warmup").exists()
#         assert (model_repository_path / "Model" / "warmup" / "input_0.data").exists


def test_input_tensor_from_metadata_return_input_tensor_when_no_batching():
    metadata = TensorMetadata()

    metadata.add(name="input_1", shape=(224, 224, 3), dtype=np.float32)
    metadata.add(name="input_2", shape=(-1, -1), dtype=np.int32)

    tensors = _input_tensor_from_metadata(input_metadata=metadata, batching=False)

    assert len(tensors) == 2

    assert tensors[0].name == "input_1"
    assert tensors[0].shape == (224, 224, 3)
    assert tensors[0].dtype == np.dtype(np.float32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].optional is False
    assert tensors[0].format is None
    assert tensors[0].allow_ragged_batch is False

    assert tensors[1].name == "input_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.int32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].optional is False
    assert tensors[1].format is None
    assert tensors[1].allow_ragged_batch is False


def test_input_tensor_from_metadata_return_input_tensor_when_batching():
    metadata = TensorMetadata()

    metadata.add(name="input_1", shape=(-1, 224, 224, 3), dtype=np.float32)
    metadata.add(name="input_2", shape=(-1, -1, -1), dtype=np.int32)

    tensors = _input_tensor_from_metadata(input_metadata=metadata, batching=True)

    assert len(tensors) == 2

    assert tensors[0].name == "input_1"
    assert tensors[0].shape == (224, 224, 3)
    assert tensors[0].dtype == np.dtype(np.float32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].optional is False
    assert tensors[0].format is None
    assert tensors[0].allow_ragged_batch is False

    assert tensors[1].name == "input_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.int32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].optional is False
    assert tensors[1].format is None
    assert tensors[1].allow_ragged_batch is False


def test_output_tensor_from_metadata_return_input_tensor_when_no_batching():
    metadata = TensorMetadata()

    metadata.add(name="output_1", shape=(1000,), dtype=np.int32)
    metadata.add(name="output_2", shape=(-1, -1), dtype=np.float32)

    tensors = _output_tensor_from_metadata(output_metadata=metadata, batching=False)

    assert len(tensors) == 2

    assert tensors[0].name == "output_1"
    assert tensors[0].shape == (1000,)
    assert tensors[0].dtype == np.dtype(np.int32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].label_filename is None

    assert tensors[1].name == "output_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.float32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].label_filename is None


def test_output_tensor_from_metadata_return_input_tensor_when_batching():
    metadata = TensorMetadata()

    metadata.add(name="output_1", shape=(-1, 1000), dtype=np.int32)
    metadata.add(name="output_2", shape=(-1, -1, -1), dtype=np.float32)

    tensors = _output_tensor_from_metadata(output_metadata=metadata, batching=True)

    assert len(tensors) == 2

    assert tensors[0].name == "output_1"
    assert tensors[0].shape == (1000,)
    assert tensors[0].dtype == np.dtype(np.int32)
    assert tensors[0].reshape == ()
    assert tensors[0].is_shape_tensor is False
    assert tensors[0].label_filename is None

    assert tensors[1].name == "output_2"
    assert tensors[1].shape == (-1, -1)
    assert tensors[1].dtype == np.dtype(np.float32)
    assert tensors[1].reshape == ()
    assert tensors[1].is_shape_tensor is False
    assert tensors[1].label_filename is None
