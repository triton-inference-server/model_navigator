# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import logging
from distutils.version import LooseVersion
from pathlib import Path
from typing import Optional, Union

from google.protobuf import text_format  # pytype: disable=pyi-error
from google.protobuf.text_format import MessageToString  # pytype: disable=pyi-error

from model_navigator.exceptions import ModelNavigatorDeployerException
from model_navigator.model import Format, Model, ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.triton.client import client_utils, grpc_client
from model_navigator.triton.config import (
    BackendAccelerator,
    DeviceKind,
    TensorRTOptPrecision,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
    TritonModelSchedulerConfig,
)

LOGGER = logging.getLogger(__name__)

_PLATFORM_PER_FORMAT = {
    Format.TF_SAVEDMODEL: "tensorflow_savedmodel",
    Format.ONNX: "onnxruntime_onnx",
    Format.TENSORRT: "tensorrt_plan",
    Format.TORCHSCRIPT: "pytorch_libtorch",
}

INT_DATATYPE2STR_DATATYPE = {
    getattr(grpc_client.model_config_pb2, attr_name): attr_name.split("_")[1]  # remove TYPE_ prefix
    for attr_name in vars(grpc_client.model_config_pb2)
    if attr_name.startswith("TYPE_")
}


class TritonModelConfigGenerator:
    def __init__(
        self,
        model: Model,
        optimization_config: TritonModelOptimizationConfig,
        scheduler_config: TritonModelSchedulerConfig,
        instances_config: TritonModelInstancesConfig,
        target_triton_version: Optional[str] = None,
    ):
        self._model = model
        self._optimization_config = optimization_config
        self._scheduler_config = scheduler_config
        self._instances_config = instances_config
        self._target_triton_version = target_triton_version
        self._validate()

    @property
    def model(self):
        return self._model

    @property
    def optimization_config(self):
        return self._optimization_config

    @property
    def scheduler_config(self):
        return self._scheduler_config

    @property
    def instances_config(self):
        return self._instances_config

    def _validate(self):

        platform = _PLATFORM_PER_FORMAT[self._model.format]

        if self._optimization_config.backend_accelerator == BackendAccelerator.AMP and not platform.startswith(
            "tensorflow"
        ):
            raise ValueError("AMP acceleration is available only for TensorFlow backends")

        if self._optimization_config.backend_accelerator == BackendAccelerator.TRT and not (
            platform.startswith("onnx") or platform.startswith("tensorflow")
        ):
            raise ValueError("TensorRT acceleration is available only for ONNX and TensorFlow backends")

    def generate_prototxt_payload(self):

        # https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/protobuf_api/model_config.proto.html
        model_config = grpc_client.model_config_pb2.ModelConfig()
        model_config.name = self._model.name
        model_config.platform = _PLATFORM_PER_FORMAT[self._model.format]

        self._extract_signature(model_config)
        self._fill_optimization(model_config)
        self._fill_scheduler(model_config)
        self._fill_instance_group(model_config)

        config_payload = MessageToString(model_config)
        LOGGER.debug(f"Generated Triton config:\n{config_payload}")

        return config_payload

    def _fill_instance_group(self, model_config):
        for kind, count in self._instances_config.engine_count_per_device.items():
            instance_group = model_config.instance_group.add()
            instance_group.kind = {
                DeviceKind.CPU: instance_group.KIND_CPU,
                DeviceKind.GPU: instance_group.KIND_GPU,
            }[kind]
            instance_group.count = count

    def _fill_scheduler(self, model_config):
        model_config.max_batch_size = self._scheduler_config.max_batch_size
        if any([self._scheduler_config.preferred_batch_sizes, self._scheduler_config.max_queue_delay_us > 0]):
            model_support_batching = self._scheduler_config.max_batch_size > 0
            if model_support_batching:
                model_config.dynamic_batching.max_queue_delay_microseconds = max(
                    int(self._scheduler_config.max_queue_delay_us), 0
                )
                preferred_batch_sizes = self._scheduler_config.preferred_batch_sizes or [
                    self._scheduler_config.max_batch_size
                ]
                for preferred_batch_size in preferred_batch_sizes:
                    model_config.dynamic_batching.preferred_batch_size.append(int(preferred_batch_size))
            else:
                LOGGER.warning("Ignore dynamic batching parameters as model doesn't support batching")

    def _fill_optimization(self, model_config):
        if self._optimization_config.backend_accelerator == BackendAccelerator.TRT:
            accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "tensorrt"
            accelerator.parameters["precision_mode"] = self._optimization_config.tensorrt_precision.value.upper()
        elif self._optimization_config.backend_accelerator == BackendAccelerator.AMP:
            accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "auto_mixed_precision"
        if model_config.platform == "tensorrt_plan":
            model_config.optimization.cuda.graphs = int(self._optimization_config.tensorrt_capture_cuda_graph)

    def _extract_signature(self, model_config):
        platform = _PLATFORM_PER_FORMAT[self._model.format]
        if self._should_extract_signature(platform):
            if not self._model.signature.inputs or not self._model.signature.outputs:
                raise ModelNavigatorDeployerException(
                    f"For {self._model.path} model, signature is required to create Triton Model Configuration. "
                    f"Could not obtain it."
                )

            def _rewrite_io_spec(spec_, item):
                dtype = f"TYPE_{client_utils.np_to_triton_dtype(spec_.dtype)}"
                dims = [1] if len(spec_.shape) <= 1 else spec_.shape[1:]  # do not pass batch size

                item.name = spec_.name
                item.dims.extend(dims)
                item.data_type = getattr(grpc_client.model_config_pb2, dtype)
                if len(spec_.shape) <= 1:
                    item.reshape.shape.extend([])

            for _name, spec in self._model.signature.inputs.items():
                input_item = model_config.input.add()
                _rewrite_io_spec(spec, input_item)

            for _name, spec in self._model.signature.outputs.items():
                output_item = model_config.output.add()
                _rewrite_io_spec(spec, output_item)

    def _should_extract_signature(self, platform):
        # https://github.com/triton-inference-server/onnxruntime_backend/pull/16
        TRITON_VERSION_WITH_FIXED_ONNX_SIGNATURE_EXTRACT = LooseVersion("2.7.0")
        version_of_triton_with_bug = (
            self._target_triton_version is not None
            and LooseVersion(self._target_triton_version) < TRITON_VERSION_WITH_FIXED_ONNX_SIGNATURE_EXTRACT
        )
        return platform.startswith("pytorch_") or (version_of_triton_with_bug and platform.startswith("onnx"))

    def save(self, model_dir: Union[str, Path]) -> str:
        """
        Serialize ModelConfig to prototxt and save to model_dir directory.
        model_dir: Union[str, Path] - path to directory where configuration will be stored
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        config_path = model_dir / "config.pbtxt"
        config_payload = self.save_config_pbtxt(config_path)
        return config_payload

    def save_config_pbtxt(self, config_path):
        config_payload = self.generate_prototxt_payload()
        with config_path.open("w+") as cfg:
            cfg.write(config_payload)
        return config_payload

    @classmethod
    def from_triton_config_pbtxt(cls, config_path: Path, model_path: Optional[Path] = None):
        with config_path.open("r") as config_file:
            payload = config_file.read()
            model_config = text_format.Parse(payload, grpc_client.model_config_pb2.ModelConfig())

        model_name = model_config.name
        # there should be exactly one version dir
        model_path = model_path or config_path.parent
        version_dir_path = [file_path for file_path in model_path.iterdir() if file_path.is_dir()][0]
        model_paths = list(version_dir_path.iterdir())
        assert len(model_paths) == 1
        model_path = model_paths[0].resolve()

        optimization_config_kwargs = {}
        if model_config.optimization.execution_accelerators.gpu_execution_accelerator:
            assert len(model_config.optimization.execution_accelerators.gpu_execution_accelerator) == 1
            backend_accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator[0]
            BACKEND_ACCELERATOR_NAMES2ACCELERATORS = {
                "tensorrt": BackendAccelerator.TRT,
                "auto_mixed_precision": BackendAccelerator.AMP,
            }
            precision_mode = backend_accelerator.parameters.get("precision_mode")
            optimization_config_kwargs = {
                "backend_accelerator": BACKEND_ACCELERATOR_NAMES2ACCELERATORS[backend_accelerator.name],
                "tensorrt_precision": TensorRTOptPrecision(precision_mode.lower()) if precision_mode else None,
            }

        optimization_config = TritonModelOptimizationConfig(
            **optimization_config_kwargs,
            tensorrt_capture_cuda_graph=(
                model_config.platform == "tensorrt_plan" and model_config.optimization.cuda.graphs
            ),
        )

        scheduler_config = TritonModelSchedulerConfig(
            max_batch_size=model_config.max_batch_size,
            preferred_batch_sizes=list(model_config.dynamic_batching.preferred_batch_size) or None,
            max_queue_delay_us=model_config.dynamic_batching.max_queue_delay_microseconds,
        )

        KIND_MAP = {
            grpc_client.model_config_pb2.ModelInstanceGroup.KIND_CPU: DeviceKind.CPU,
            grpc_client.model_config_pb2.ModelInstanceGroup.KIND_GPU: DeviceKind.GPU,
        }

        instances_config = TritonModelInstancesConfig(
            engine_count_per_device={KIND_MAP[entry.kind]: entry.count for entry in model_config.instance_group}
        )

        model = Model(model_name, model_path)
        # if there is no possibility to obtain signature (also from annotation file)
        # try to load it from triton config file
        if model.signature is not None and model.signature.is_missing():

            def _rewrite_io_spec(item):
                data_type = INT_DATATYPE2STR_DATATYPE[item.data_type]
                np_class = client_utils.triton_to_np_dtype(data_type)
                dummy_instance = np_class()
                dtype = dummy_instance.dtype
                if len(item.dims) == 0:
                    shape = (-1,)
                else:
                    shape = (-1,) + tuple(item.dims)
                return TensorSpec(name=item.name, shape=shape, dtype=dtype)

            inputs = {item.name: _rewrite_io_spec(item) for item in model_config.input}
            outputs = {item.name: _rewrite_io_spec(item) for item in model_config.output}
            signature = ModelSignatureConfig(inputs=inputs, outputs=outputs)
            model = Model(model_name, model_path, signature_if_missing=signature)

        return cls(
            model=model,
            optimization_config=optimization_config,
            scheduler_config=scheduler_config,
            instances_config=instances_config,
        )
