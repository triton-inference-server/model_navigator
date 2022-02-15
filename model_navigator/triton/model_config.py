# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from pathlib import Path
from typing import Optional, Union

from google.protobuf import text_format  # pytype: disable=pyi-error
from google.protobuf.text_format import MessageToString  # pytype: disable=pyi-error
from tritonclient.grpc import model_config_pb2

from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.model import Model, ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.triton.backends.base import BackendConfiguratorSelector
from model_navigator.triton.client import client_utils, grpc_client
from model_navigator.triton.config import (
    BackendAccelerator,
    Batching,
    DeviceKind,
    TensorRTOptPrecision,
    TritonBatchingConfig,
    TritonCustomBackendParametersConfig,
    TritonDynamicBatchingConfig,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
)

LOGGER = logging.getLogger(__name__)


class ModelConfigParser:
    @classmethod
    def parse(cls, *, config_path: Path, external_model_path: Optional[Path] = None, config_cls):
        INT_DATATYPE2STR_DATATYPE = {
            getattr(model_config_pb2, attr_name): attr_name.split("_")[1]  # remove TYPE_ prefix
            for attr_name in vars(model_config_pb2)
            if attr_name.startswith("TYPE_")
        }

        LOGGER.debug(f"Parsing Triton config model config_path={config_path} external_model_path={external_model_path}")

        with config_path.open("r") as config_file:
            payload = config_file.read()
            model_config = text_format.Parse(payload, model_config_pb2.ModelConfig())

        model_name = model_config.name

        if not external_model_path:
            from model_navigator.triton import TritonModelStore

            model_store = TritonModelStore(config_path.parent.parent)
            model_path = model_store.get_model_path(config_path.parent.name)
        else:
            model_path = external_model_path
        model_path = Path(model_path)

        optimization_config_kwargs = {}
        tensorrt_common_config_kwargs = {}
        if model_config.optimization.execution_accelerators.gpu_execution_accelerator:
            assert len(model_config.optimization.execution_accelerators.gpu_execution_accelerator) == 1
            backend_accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator[0]
            BACKEND_ACCELERATOR_NAMES2ACCELERATORS = {
                "tensorrt": BackendAccelerator.TRT,
                "auto_mixed_precision": BackendAccelerator.AMP,
            }
            precision_mode = backend_accelerator.parameters.get("precision_mode")
            max_workspace_size = backend_accelerator.parameters.get("max_workspace_size")

            optimization_config_kwargs = {
                "backend_accelerator": BACKEND_ACCELERATOR_NAMES2ACCELERATORS[backend_accelerator.name],
                "tensorrt_precision": TensorRTOptPrecision(precision_mode.lower()) if precision_mode else None,
            }

            tensorrt_common_config_kwargs = {
                "tensorrt_max_workspace_size": max_workspace_size if max_workspace_size else None
            }

        if model_config.dynamic_batching.preferred_batch_size:
            batching = Batching.DYNAMIC
        elif model_config.max_batch_size > 0:
            batching = Batching.STATIC
        else:
            batching = Batching.DISABLED

        batching_config = TritonBatchingConfig(
            max_batch_size=model_config.max_batch_size,
            batching=batching,
        )
        optimization_config = TritonModelOptimizationConfig(
            **optimization_config_kwargs,
            tensorrt_capture_cuda_graph=(
                model_config.platform == "tensorrt_plan" and model_config.optimization.cuda.graphs
            ),
        )

        tensorrt_common_config = TensorRTCommonConfig(**tensorrt_common_config_kwargs)

        dynamic_batching_config = TritonDynamicBatchingConfig(
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

        backend_parameters_config = TritonCustomBackendParametersConfig(
            triton_backend_parameters={
                name: model_config.parameters[name].string_value for name in model_config.parameters
            }
        )

        explicit_format = None
        model = Model(model_name, model_path, explicit_format=explicit_format)
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
            model = Model(model_name, model_path, signature_if_missing=signature, explicit_format=explicit_format)

        return config_cls(
            model=model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            dynamic_batching_config=dynamic_batching_config,
            tensorrt_common_config=tensorrt_common_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )


class TritonModelConfigGenerator:
    def __init__(
        self,
        model: Model,
        *,
        batching_config: TritonBatchingConfig,
        optimization_config: TritonModelOptimizationConfig,
        tensorrt_common_config: TensorRTCommonConfig,
        dynamic_batching_config: TritonDynamicBatchingConfig,
        instances_config: TritonModelInstancesConfig,
        backend_parameters_config: TritonCustomBackendParametersConfig,
        target_triton_version: Optional[str] = None,
    ):
        self._model = model
        self._batching_config = batching_config
        self._optimization_config = optimization_config
        self._tensorrt_common_config = tensorrt_common_config
        self._dynamic_batching_config = dynamic_batching_config
        self._instances_config = instances_config
        self._backend_parameters_config = backend_parameters_config
        self._target_triton_version = target_triton_version

    @property
    def model(self):
        return self._model

    @property
    def batching_config(self):
        return self._batching_config

    @property
    def optimization_config(self):
        return self._optimization_config

    @property
    def tensorrt_common_config(self):
        return self._tensorrt_common_config

    @property
    def dynamic_batching_config(self):
        return self._dynamic_batching_config

    @property
    def instances_config(self):
        return self._instances_config

    @property
    def backend_parameters_config(self):
        return self._backend_parameters_config

    def save(self, config_path: Union[str, Path]) -> str:
        """
        Serialize ModelConfig to prototxt and save to config_path directory.
        config_path: Union[str, Path] - path to configuration file
        """

        # https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
        model_config = grpc_client.model_config_pb2.ModelConfig()
        backend_configurator = BackendConfiguratorSelector.for_model(self._model)
        backend_configurator.update_config_for_model(
            model_config,
            self._model,
            batching_config=self._batching_config,
            optimization_config=self._optimization_config,
            tensorrt_common_config=self._tensorrt_common_config,
            dynamic_batching_config=self._dynamic_batching_config,
            instances_config=self._instances_config,
            backend_parameters_config=self._backend_parameters_config,
        )

        config_payload = MessageToString(model_config)
        LOGGER.debug(f"Generated Triton config:\n{config_payload}")

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w+") as cfg:
            cfg.write(config_payload)
        return config_payload

    @classmethod
    def parse_triton_config_pbtxt(cls, config_path: Path, external_model_path: Optional[Path] = None):
        return ModelConfigParser.parse(config_path=config_path, external_model_path=external_model_path, config_cls=cls)
