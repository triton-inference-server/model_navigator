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
"""Triton Model Store generator class module.

Example of use:
    triton_model_repository = TritonModelRepository(model_repository_path="/path/to/model/store")

    triton_model_repository.deploy_model(
        model_path="/path/to/serialized/model.onnx",
        model_config=ModelConfig(
            model_name="ONNXModel",
            backend=Backend.ONNX
        )
    )
"""
import logging
import pathlib
import shutil
from typing import List, Optional, Union

from model_navigator.api.config import Format
from model_navigator.commands.performance import Performance
from model_navigator.exceptions import (
    ModelNavigatorEmptyPackageError,
    ModelNavigatorError,
    ModelNavigatorWrongParameterError,
)
from model_navigator.frameworks import is_tf_available, is_torch_available  # noqa: F401
from model_navigator.package.package import Package
from model_navigator.runners.onnx import OnnxrtCPURunner, OnnxrtCUDARunner, OnnxrtTensorRTRunner
from model_navigator.runners.tensorflow import (
    TensorFlowSavedModelCPURunner,
    TensorFlowSavedModelCUDARunner,
    TensorFlowTensorRTRunner,
)
from model_navigator.runners.tensorrt import TensorRTRunner
from model_navigator.runners.torch import TorchScriptCPURunner, TorchScriptCUDARunner, TorchTensorRTRunner
from model_navigator.runtime_analyzer import RuntimeAnalyzer
from model_navigator.runtime_analyzer.analyzer import RuntimeAnalyzerResult
from model_navigator.runtime_analyzer.strategy import MaxThroughputAndMinLatencyStrategy, RuntimeSearchStrategy
from model_navigator.triton.model_config import ModelConfig
from model_navigator.triton.model_config_builder import ModelConfigBuilder
from model_navigator.triton.model_config_generator import ModelConfigGenerator
from model_navigator.triton.specialized_configs import (
    Backend,
    DeviceKind,
    InputTensorSpec,
    InstanceGroup,
    ONNXModelConfig,
    ONNXOptimization,
    OutputTensorSpec,
    PythonModelConfig,
    PyTorchModelConfig,
    TensorFlowModelConfig,
    TensorRTAccelerator,
    TensorRTModelConfig,
)
from model_navigator.triton.utils import input_tensor_from_metadata, output_tensor_from_metadata

LOGGER = logging.getLogger(__name__)

BACKEND2SUFFIX = {
    Backend.ONNXRuntime: ".onnx",
    Backend.Python: ".py",
    Backend.PyTorch: ".pt",
    Backend.TensorFlow: ".savedmodel",
    Backend.TensorRT: ".plan",
}

TRITON_FORMATS = (
    Format.ONNX,
    Format.TF_TRT,
    Format.TORCH_TRT,
    Format.TORCHSCRIPT,
    Format.TENSORRT,
    Format.TF_SAVEDMODEL,
)

TRITON_RUNNERS = (
    OnnxrtCPURunner,
    OnnxrtCUDARunner,
    OnnxrtTensorRTRunner,
    TensorRTRunner,
    TorchScriptCUDARunner,
    TorchScriptCPURunner,
    TorchTensorRTRunner,
    TensorFlowSavedModelCPURunner,
    TensorFlowSavedModelCUDARunner,
    TensorFlowTensorRTRunner,
)


def add_model(
    model_repository_path: Union[str, pathlib.Path],
    model_name: str,
    model_path: Union[str, pathlib.Path],
    config: Union[
        ONNXModelConfig,
        TensorRTModelConfig,
        PyTorchModelConfig,
        PythonModelConfig,
        TensorFlowModelConfig,
    ],
    model_version: int = 1,
) -> pathlib.Path:
    """Generate model deployment inside provided model store path.

    The config requires specialized configuration to be passed for backend on which model is executed. Example:

    - ONNX model requires ONNXModelConfig
    - TensorRT model requires TensorRTModelConfig
    - TorchScript or Torch-TensorRT models requires PyTorchModelConfig
    - TensorFlow SavedModel or TensorFlow-TensorRT models requires TensorFlowModelConfig
    - Python model requires PythonModelConfig

    Args:
        model_repository_path: Path where deployment should be created
        model_name: Name under which model is deployed in Triton Inference Server
        model_path: Path to model
        config: Specialized configuration of model for backend on which model is executed
        model_version: Version of model that is deployed

    Returns:
         Path to created model store
    """
    if isinstance(config, ONNXModelConfig):
        model_config = ModelConfigBuilder.from_onnx_config(
            model_name=model_name,
            model_version=model_version,
            onnx_config=config,
        )
    elif isinstance(config, TensorFlowModelConfig):
        model_config = ModelConfigBuilder.from_tensorflow_config(
            model_name=model_name,
            model_version=model_version,
            tensorflow_config=config,
        )
    elif isinstance(config, PythonModelConfig):
        model_config = ModelConfigBuilder.from_python_config(
            model_name=model_name,
            model_version=model_version,
            python_config=config,
        )
    elif isinstance(config, PyTorchModelConfig):
        model_config = ModelConfigBuilder.from_pytorch_config(
            model_name=model_name,
            model_version=model_version,
            pytorch_config=config,
        )
    elif isinstance(config, TensorRTModelConfig):
        model_config = ModelConfigBuilder.from_tensorrt_config(
            model_name=model_name,
            model_version=model_version,
            tensorrt_config=config,
        )
    else:
        raise ModelNavigatorWrongParameterError(f"Unsupported model config provided: {config.__class__}")

    triton_model_repository = _TritonModelRepository(model_repository_path=pathlib.Path(model_repository_path))
    return triton_model_repository.deploy_model(
        model_path=pathlib.Path(model_path),
        model_config=model_config,
    )


def add_model_from_package(
    model_repository_path: Union[str, pathlib.Path],
    model_name: str,
    package: Package,
    model_version: int = 1,
    strategy: Optional[RuntimeSearchStrategy] = None,
    response_cache: bool = False,
):
    """Create the Triton Model Store with optimized model and save it to `model_repository_path`.

    Args:
        model_repository_path: Path where the model store is located
        model_name: Name under which model is deployed in Triton Inference Server
        model_version: Version of model that is deployed
        package: Package for which model store is created
        strategy: Strategy for finding the best runtime.
                  When not set the `MaxThroughputAndMinLatencyStrategy` is used.
        response_cache: Enable response cache for model

    Returns:
        Path to created model store
    """
    if package.is_empty():
        raise ModelNavigatorEmptyPackageError("No models available in the package. Triton deployment is not possible.")

    if package.config.batch_dim not in [0, None]:
        raise ModelNavigatorWrongParameterError(
            "Only models without batching or batch dimension on first place in shape are supported for Triton."
        )

    if strategy is None:
        strategy = MaxThroughputAndMinLatencyStrategy()

    batching = package.config.batch_dim == 0

    runtime_result = RuntimeAnalyzer.get_runtime(
        models_status=package.status.models_status,
        strategy=strategy,
        formats=[fmt.value for fmt in TRITON_FORMATS],
        runners=[runner.name() for runner in TRITON_RUNNERS],
    )
    max_batch_size = max(
        profiling_results.batch_size
        for profiling_results in runtime_result.runner_status.result[Performance.name]["profiling_results"]
    )

    if runtime_result.model_status.model_config.format == Format.ONNX:
        config = _onnx_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            response_cache=response_cache,
            runtime_result=runtime_result,
        )

    elif runtime_result.model_status.model_config.format in [Format.TF_SAVEDMODEL, Format.TF_TRT]:
        config = _tensorflow_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            response_cache=response_cache,
            runtime_result=runtime_result,
        )
    elif runtime_result.model_status.model_config.format in [Format.TORCHSCRIPT, Format.TORCH_TRT]:
        inputs = input_tensor_from_metadata(
            package.status.input_metadata,
            batching=batching,
        )
        outputs = output_tensor_from_metadata(
            package.status.output_metadata,
            batching=batching,
        )

        config = _pytorch_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            inputs=inputs,
            outputs=outputs,
            response_cache=response_cache,
            runtime_result=runtime_result,
        )
    elif runtime_result.model_status.model_config.format == Format.TENSORRT:
        config = _tensorrt_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            response_cache=response_cache,
        )
    else:
        raise ModelNavigatorError(
            f"Unsupported model format selected: {runtime_result.model_status.model_config.format}"
        )

    return add_model(
        model_repository_path=model_repository_path,
        model_name=model_name,
        model_version=model_version,
        model_path=package.workspace.path / runtime_result.model_status.model_config.path,
        config=config,
    )


class _TritonModelRepository:
    """Class for deploying models inside the Triton Model Store."""

    def __init__(self, model_repository_path: pathlib.Path):
        """Initialize model repository object."""
        self._model_repository_path = model_repository_path

    def deploy_model(
        self,
        *,
        model_path: pathlib.Path,
        model_config: ModelConfig,
    ) -> pathlib.Path:
        """Deploy model with provided config to model store.

        Args:
            model_path: Path to model that has to be deployed to model store
            model_config: Configuration of model deployment

        Returns:
            Path to deployed model inside the model store
        """
        LOGGER.info(
            f"Deploying model {model_path} of version {model_config.model_version} in "
            f"Triton Model Store {self._model_repository_path}/{model_config.model_name}"
        )

        # Order of model repository files might be important while using Triton server in polling model_control_mode
        model_path = self._copy_model(
            model_path=model_path,
            backend=model_config.backend or model_config.platform,
            model_name=model_config.model_name,
            version=model_config.model_version,
        )

        # remove model filename and model version
        model_dir_in_model_repository_path = model_path.parent.parent
        config_path = model_dir_in_model_repository_path / "config.pbtxt"

        generator = ModelConfigGenerator(config=model_config)
        generator.to_file(config_path=config_path)

        return model_dir_in_model_repository_path

    def _copy_model(
        self,
        *,
        model_path: pathlib.Path,
        backend: Backend,
        model_name: str,
        version: int,
    ) -> pathlib.Path:
        dst_path = self._get_model_path(
            model_name=model_name,
            version=version,
            backend=backend,
        )
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        LOGGER.debug(f"Copying {model_path} to {dst_path}")
        if model_path.is_file():
            shutil.copy(model_path, dst_path)
        else:
            try:
                shutil.copytree(model_path, dst_path)
            except shutil.Error:
                # due to error as reported on https://bugs.python.org/issue43743
                shutil._USE_CP_SENDFILE = False
                shutil.rmtree(dst_path)
                shutil.copytree(model_path, dst_path)
        return dst_path

    def _get_model_path(
        self,
        *,
        model_name: str,
        version: int,
        backend: Backend,
    ) -> pathlib.Path:
        return self._model_repository_path / model_name / str(version) / self._get_filename(backend=backend)

    def _get_filename(self, *, backend: Backend):
        suffix = BACKEND2SUFFIX[backend]
        return f"model{suffix}"


def _onnx_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
    runtime_result: RuntimeAnalyzerResult,
):
    optimization = None
    instance_groups = []
    if runtime_result.runner_status.runner_name == OnnxrtTensorRTRunner.name():
        optimization = ONNXOptimization(accelerator=TensorRTAccelerator())

    if runtime_result.runner_status.runner_name in [OnnxrtCUDARunner.name(), OnnxrtTensorRTRunner.name()]:
        instance_groups = [InstanceGroup(kind=DeviceKind.KIND_GPU)]

    config = ONNXModelConfig(
        batching=batching,
        max_batch_size=max_batch_size,
        response_cache=response_cache,
        optimization=optimization,
        instance_groups=instance_groups,
    )
    return config


def _tensorflow_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
    runtime_result: RuntimeAnalyzerResult,
):
    instance_groups = []
    if runtime_result.runner_status.runner_name == TensorFlowTensorRTRunner.name():
        instance_groups = [InstanceGroup(kind=DeviceKind.KIND_GPU)]

    # TODO: check runner for savedmodel when available

    config = TensorFlowModelConfig(
        batching=batching,
        max_batch_size=max_batch_size,
        response_cache=response_cache,
        instance_groups=instance_groups,
    )
    return config


def _pytorch_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
    inputs: List[InputTensorSpec],
    outputs: List[OutputTensorSpec],
    runtime_result: RuntimeAnalyzerResult,
):
    instance_groups = []
    if runtime_result.runner_status.runner_name in [TorchTensorRTRunner.name(), TorchScriptCUDARunner.name()]:
        instance_groups = [InstanceGroup(kind=DeviceKind.KIND_GPU)]

    config = PyTorchModelConfig(
        batching=batching,
        max_batch_size=max_batch_size,
        inputs=inputs,
        outputs=outputs,
        response_cache=response_cache,
        instance_groups=instance_groups,
    )
    return config


def _tensorrt_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
):
    config = TensorRTModelConfig(
        batching=batching,
        max_batch_size=max_batch_size,
        response_cache=response_cache,
        instance_groups=[InstanceGroup(kind=DeviceKind.KIND_GPU)],
    )
    return config
