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
from typing import Dict, List, Optional, Union

import numpy as np

from model_navigator.api.config import Format, Sample, TensorRTProfile
from model_navigator.commands.performance import Performance
from model_navigator.core.dataloader import load_samples
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import (
    ModelNavigatorEmptyPackageError,
    ModelNavigatorError,
    ModelNavigatorWrongParameterError,
)
from model_navigator.frameworks import is_tf_available, is_torch_available  # noqa: F401
from model_navigator.frameworks.tensorrt import utils as trt_utils
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
    ModelWarmup,
    ModelWarmupInput,
    ModelWarmupInputDataType,
    ONNXModelConfig,
    ONNXOptimization,
    OutputTensorSpec,
    PythonModelConfig,
    PyTorchModelConfig,
    SequenceBatcher,
    TensorFlowModelConfig,
    TensorRTAccelerator,
    TensorRTModelConfig,
)

LOGGER = logging.getLogger(__name__)

BACKEND2SUFFIX = {
    Backend.ONNXRuntime: ".onnx",
    Backend.Python: ".py",
    Backend.PyTorch: ".pt",
    Backend.TensorFlow: ".savedmodel",
    Backend.TensorRT: ".plan",
}

BACKEND2CATALOGBASEDMODEL = {
    Backend.PyTorch: True,
    Backend.TensorFlow: True,
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

    initial_state_files = _collect_initial_state_files(model_config=model_config)
    warmup_files = _collect_warmup_files(model_config=model_config)

    triton_model_repository = _TritonModelRepository(model_repository_path=pathlib.Path(model_repository_path))
    return triton_model_repository.deploy_model(
        model_path=pathlib.Path(model_path),
        model_config=model_config,
        warmup_files=warmup_files,
        initial_state_files=initial_state_files,
    )


def add_model_from_package(
    model_repository_path: Union[str, pathlib.Path],
    model_name: str,
    package: Package,
    model_version: int = 1,
    strategy: Optional[RuntimeSearchStrategy] = None,
    response_cache: bool = False,
    warmup: bool = False,
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
        warmup: Enable warmup for min and max batch size

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
        profiling_results.batch_size if profiling_results.batch_size is not None else 0
        for profiling_results in runtime_result.runner_status.result[Performance.name]["profiling_results"]
    )

    model_warmup = {}
    if warmup:
        model_warmup = _prepare_model_warmup(
            batching=batching,
            max_batch_size=max_batch_size,
            package=package,
        )

    if runtime_result.model_status.model_config.format == Format.TENSORRT:
        input_metadata, output_metadata = (
            _prepare_tensorrt_metadata(package.status.input_metadata),
            _prepare_tensorrt_metadata(package.status.output_metadata),
        )
    else:
        input_metadata, output_metadata = package.status.input_metadata, package.status.output_metadata

    inputs = _input_tensor_from_metadata(
        input_metadata,
        batching=batching,
    )
    outputs = _output_tensor_from_metadata(
        output_metadata,
        batching=batching,
    )

    if runtime_result.model_status.model_config.format == Format.ONNX:
        config = _onnx_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            inputs=inputs,
            outputs=outputs,
            response_cache=response_cache,
            runtime_result=runtime_result,
            warmup=model_warmup,
        )

    elif runtime_result.model_status.model_config.format in [Format.TF_SAVEDMODEL, Format.TF_TRT]:
        config = _tensorflow_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            inputs=inputs,
            outputs=outputs,
            response_cache=response_cache,
            runtime_result=runtime_result,
            warmup=model_warmup,
        )
    elif runtime_result.model_status.model_config.format in [Format.TORCHSCRIPT, Format.TORCH_TRT]:
        config = _pytorch_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            inputs=inputs,
            outputs=outputs,
            response_cache=response_cache,
            runtime_result=runtime_result,
            warmup=model_warmup,
        )
    elif runtime_result.model_status.model_config.format == Format.TENSORRT:
        config = _tensorrt_config_from_runtime_result(
            batching=batching,
            max_batch_size=max_batch_size,
            inputs=inputs,
            outputs=outputs,
            response_cache=response_cache,
            warmup=model_warmup,
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


def _collect_warmup_files(model_config: ModelConfig):
    warmup_files = []
    for warmup in model_config.warmup.values():
        for inpt in warmup.inputs.values():
            if inpt.input_data_file:
                warmup_files.append(inpt.input_data_file)

    return warmup_files


def _collect_initial_state_files(model_config: ModelConfig):
    initial_state_files = []
    if isinstance(model_config.batcher, SequenceBatcher):
        for state in model_config.batcher.states:
            for initial_state in state.initial_states:
                if initial_state.data_file:
                    initial_state_files.append(initial_state.data_file)

    return initial_state_files


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
        warmup_files: List[pathlib.Path],
        initial_state_files: List[pathlib.Path],
    ) -> pathlib.Path:
        """Deploy model with provided config to model store.

        Args:
            model_path: Path to model that has to be deployed to model store
            model_config: Configuration of model deployment
            warmup_files: List of warmup files to copy
            initial_state_files: List of initial states files to copy

        Returns:
            Path to deployed model inside the model store
        """
        LOGGER.info(
            f"Deploying model {model_path} of version {model_config.model_version} in "
            f"Triton Model Store {self._model_repository_path}/{model_config.model_name}"
        )
        # Collect model filename if default not provided
        backend = model_config.backend or model_config.platform
        model_filename = model_config.default_model_filename or self._get_default_filename(backend=backend)

        # Path to model version catalog
        model_version_path = self._get_version_path(
            model_name=model_config.model_name, version=model_config.model_version
        )

        # Order of model repository files might be important while using Triton server in polling model_control_mode
        if model_path.is_file() or self._allow_model_catalog(backend=backend):
            self._copy_model_file(
                model_path=model_path,
                model_version_path=model_version_path,
                model_filename=model_filename,
            )
        else:
            self._copy_model_catalog(
                model_path=model_path,
                model_version_path=model_version_path,
            )

        # remove model filename and model version
        model_store_dir = model_version_path.parent

        self._copy_initial_state_files(
            model_store_dir=model_store_dir,
            initial_state_files=initial_state_files,
        )

        self._copy_warmup_files(
            model_store_dir=model_store_dir,
            warmup_files=warmup_files,
        )

        config_path = model_store_dir / "config.pbtxt"

        generator = ModelConfigGenerator(config=model_config)
        generator.to_file(config_path=config_path)

        return model_store_dir

    def _copy_model_file(
        self,
        *,
        model_path: pathlib.Path,
        model_version_path: pathlib.Path,
        model_filename: str,
    ):
        LOGGER.debug(f"Creating version directory {model_version_path}")
        model_version_path.mkdir(exist_ok=True, parents=True)
        dst_path = model_version_path / model_filename
        LOGGER.debug(f"Copying {model_path} file to {dst_path}")
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

    def _copy_model_catalog(
        self,
        *,
        model_path: pathlib.Path,
        model_version_path: pathlib.Path,
    ):
        LOGGER.debug(f"Creating model directory {model_version_path.parent}")
        model_version_path.parent.mkdir(exist_ok=True, parents=True)
        LOGGER.debug(f"Copying {model_path} file to {model_version_path}")
        try:
            shutil.copytree(model_path, model_version_path)
        except shutil.Error:
            # due to error as reported on https://bugs.python.org/issue43743
            shutil._USE_CP_SENDFILE = False
            shutil.rmtree(model_version_path)
            shutil.copytree(model_path, model_version_path)

    def _copy_initial_state_files(self, model_store_dir: pathlib.Path, initial_state_files: List[pathlib.Path]):
        if not initial_state_files:
            return

        initial_state_repository_dir = model_store_dir / "initial_state"
        initial_state_repository_dir.mkdir(exist_ok=True)

        for initial_state_file in initial_state_files:
            initial_state_file_repository_path = initial_state_repository_dir / initial_state_file.name
            shutil.copy(initial_state_file, initial_state_file_repository_path)

    def _copy_warmup_files(self, model_store_dir: pathlib.Path, warmup_files: List[pathlib.Path]):
        if not warmup_files:
            return

        warmup_repository_dir = model_store_dir / "warmup"
        warmup_repository_dir.mkdir(exist_ok=True)

        for warmup_file in warmup_files:
            warmup_file_repository_path = warmup_repository_dir / warmup_file.name
            shutil.copy(warmup_file, warmup_file_repository_path)

    def _get_version_path(self, *, model_name: str, version: int) -> pathlib.Path:
        return self._model_repository_path / model_name / str(version)

    def _get_default_filename(self, *, backend: Backend):
        suffix = BACKEND2SUFFIX[backend]
        return f"model{suffix}"

    def _allow_model_catalog(self, *, backend: Backend):
        return BACKEND2CATALOGBASEDMODEL.get(backend, False)


def _onnx_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
    inputs: List[InputTensorSpec],
    outputs: List[OutputTensorSpec],
    runtime_result: RuntimeAnalyzerResult,
    warmup: Dict[str, ModelWarmup],
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
        inputs=inputs,
        outputs=outputs,
        response_cache=response_cache,
        optimization=optimization,
        instance_groups=instance_groups,
        warmup=warmup,
    )
    return config


def _tensorflow_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
    inputs: List[InputTensorSpec],
    outputs: List[OutputTensorSpec],
    runtime_result: RuntimeAnalyzerResult,
    warmup: Dict[str, ModelWarmup],
):
    instance_groups = []
    if runtime_result.runner_status.runner_name == TensorFlowTensorRTRunner.name():
        instance_groups = [InstanceGroup(kind=DeviceKind.KIND_GPU)]

    # TODO: check runner for savedmodel when available

    config = TensorFlowModelConfig(
        batching=batching,
        max_batch_size=max_batch_size,
        inputs=inputs,
        outputs=outputs,
        response_cache=response_cache,
        instance_groups=instance_groups,
        warmup=warmup,
    )
    return config


def _pytorch_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
    inputs: List[InputTensorSpec],
    outputs: List[OutputTensorSpec],
    runtime_result: RuntimeAnalyzerResult,
    warmup: Dict[str, ModelWarmup],
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
        warmup=warmup,
    )
    return config


def _tensorrt_config_from_runtime_result(
    batching: bool,
    max_batch_size: int,
    response_cache: bool,
    inputs: List[InputTensorSpec],
    outputs: List[OutputTensorSpec],
    warmup: Dict[str, ModelWarmup],
):
    config = TensorRTModelConfig(
        batching=batching,
        max_batch_size=max_batch_size,
        inputs=inputs,
        outputs=outputs,
        response_cache=response_cache,
        instance_groups=[InstanceGroup(kind=DeviceKind.KIND_GPU)],
        warmup=warmup,
    )
    return config


def _prepare_model_warmup(max_batch_size: int, batching: bool, package: Package):
    if not batching:
        batch_sizes = [1]
    else:
        batch_sizes = {1, max_batch_size}

    profiling_sample = load_samples("profiling_sample", package.workspace.path, package.config.batch_dim)[0]

    model_warmups = {}
    for batch_size in batch_sizes:
        name = f"warmup_{batch_size}"
        model_warmups[name] = ModelWarmup(
            batch_size=batch_size,
            inputs=_warmup_input_from_metadata(
                workspace=package.workspace,
                batching=batching,
                input_metadata=package.status.input_metadata,
                trt_profile=package.status.dataloader_trt_profile,
                profiling_samples=profiling_sample,
            ),
        )

    return model_warmups


def _input_tensor_from_metadata(input_metadata: TensorMetadata, batching: bool = True) -> List:
    """Generate list of input tensors based on TensorMetadata.

    Args:
        input_metadata: Model inputs metadata
        batching: Flag indicating if input metadata contain batch in shape

    Returns:
        List of input tensors
    """
    input_tensors = []
    for metadata in input_metadata.values():
        shape = metadata.shape[1:] if batching else metadata.shape
        tensor = InputTensorSpec(name=metadata.name, dtype=metadata.dtype, shape=shape)
        input_tensors.append(tensor)
    return input_tensors


def _output_tensor_from_metadata(output_metadata: TensorMetadata, batching: bool = True) -> List:
    """Generate list of output tensors based on TensorMetadata.

    Args:
        output_metadata: Model outputs metadata
        batching: Flag indicating if output metadata contain batch in shape

    Returns:
        List of output tensors
    """
    output_tensors = []
    for metadata in output_metadata.values():
        shape = metadata.shape[1:] if batching else metadata.shape
        tensor = OutputTensorSpec(name=metadata.name, dtype=metadata.dtype, shape=shape)
        output_tensors.append(tensor)
    return output_tensors


def _warmup_input_from_metadata(
    workspace: Workspace,
    input_metadata: TensorMetadata,
    trt_profile: TensorRTProfile,
    profiling_samples: Sample,
    batching: bool = True,
) -> Dict:
    """Generate list of warmup inputs tensors based on TensorMetadata.

    Args:
        workspace: Current workspace for package
        input_metadata: Model inputs metadata
        trt_profile: TensorRT profile to get maximal shape of input
        profiling_samples: sample used for generating data for inputs
        batching: Flag indicating if output metadata contain batch in shape

    Returns:
       Dict with warmup inputs
    """
    warmup_inputs = {}
    for name, profile in trt_profile.items():
        shape = profile.max[1:] if batching else profile.max
        input_data = profiling_samples[name][0] if batching else profiling_samples[name]
        file_path = _generate_warmup_file(workspace=workspace, filename=name, input_data=input_data)
        warmup_inputs[name] = ModelWarmupInput(
            dtype=input_metadata[name].dtype,
            shape=shape,
            input_data_type=ModelWarmupInputDataType.FILE,
            input_data_file=file_path,
        )

    return warmup_inputs


def _generate_warmup_file(workspace: Workspace, filename: str, input_data: np.ndarray) -> pathlib.Path:
    """Generate warmup file with binary content in row-major order.

    Args:
        workspace: workspace where files are going to be generated
        filename: name under which the file would be generated
        input_data: data to store inside the file

    Returns:
        Path to the file
    """
    warmup_path = workspace.path / "warmup"
    warmup_path.mkdir(exist_ok=True)

    file_path = warmup_path / f"{filename}.data"
    with file_path.open("wb") as fp:
        fp.write(input_data.tobytes(order="C"))

    return file_path


def _prepare_tensorrt_metadata(metadata: TensorMetadata):
    updated_metadata = TensorMetadata()
    for name, tensor_spec in metadata.items():
        if trt_utils.cast_type(tensor_spec.dtype) != tensor_spec.dtype:
            new_dtype = trt_utils.cast_type(tensor_spec.dtype)
        else:
            new_dtype = tensor_spec.dtype
        updated_metadata.add(name, dtype=new_dtype, shape=tensor_spec.shape)
    return updated_metadata
