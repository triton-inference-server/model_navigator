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
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from google.protobuf import json_format, text_format  # pytype: disable=pyi-error

from model_navigator.common.config import BatchingConfig, TensorRTCommonConfig
from model_navigator.model import Model, ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.triton.backends.base import BackendConfiguratorSelector
from model_navigator.triton.client import client_utils
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
        from tritonclient.grpc import model_config_pb2  # pytype: disable=import-error

        LOGGER.debug(f"Parsing Triton config model config_path={config_path} external_model_path={external_model_path}")

        with config_path.open("r") as config_file:
            payload = config_file.read()
            model_config_proto = text_format.Parse(payload, model_config_pb2.ModelConfig())

        model_config_dict = json_format.MessageToDict(model_config_proto, preserving_proto_field_name=True)

        LOGGER.debug(f"Parsed model config: \n{json.dumps(model_config_dict, indent=4)}")

        model_name = model_config_dict["name"]

        if not external_model_path:
            from model_navigator.triton import TritonModelStore

            model_store = TritonModelStore(config_path.parent.parent)
            model_path = model_store.get_model_path(config_path.parent.name)
        else:
            model_path = external_model_path

        model_path = Path(model_path)

        optimization_config_kwargs = {}
        tensorrt_common_config_kwargs = {}
        gpu_execution_accelerator = cls._get_gpu_execution_accelerator(model_config_dict)
        cpu_execution_accelerator = cls._get_cpu_execution_accelerator(model_config_dict)
        if gpu_execution_accelerator:
            assert len(gpu_execution_accelerator) == 1
            backend_accelerator = gpu_execution_accelerator[0]
            BACKEND_ACCELERATOR_NAMES2ACCELERATORS = {
                "tensorrt": BackendAccelerator.TRT,
                "auto_mixed_precision": BackendAccelerator.AMP,
            }
            parameters = backend_accelerator.get("parameters", {})
            precision_mode = parameters.get("precision_mode")
            max_workspace_size = parameters.get("max_workspace_size")

            optimization_config_kwargs = {
                "backend_accelerator": BACKEND_ACCELERATOR_NAMES2ACCELERATORS[backend_accelerator["name"]],
                "tensorrt_precision": TensorRTOptPrecision(precision_mode.lower()) if precision_mode else None,
            }

            tensorrt_common_config_kwargs = {
                "tensorrt_max_workspace_size": max_workspace_size if max_workspace_size else None
            }
        elif cpu_execution_accelerator:
            assert len(cpu_execution_accelerator) == 1
            backend_accelerator = cpu_execution_accelerator[0]

            BACKEND_ACCELERATOR_NAMES2ACCELERATORS = {
                "openvino": BackendAccelerator.OPENVINO,
            }

            optimization_config_kwargs = {
                "backend_accelerator": BACKEND_ACCELERATOR_NAMES2ACCELERATORS[backend_accelerator["name"]],
            }

        if "dynamic_batching" in model_config_dict:
            batching = Batching.DYNAMIC
        elif model_config_dict.get("max_batch_size", 0) > 0:
            batching = Batching.STATIC
        else:
            batching = Batching.DISABLED

        batching_config = BatchingConfig(
            max_batch_size=model_config_dict.get("max_batch_size", 0),
        )

        triton_batching_config = TritonBatchingConfig(
            batching=batching,
        )

        if "dynamic_batching" in model_config_dict:
            dynamic_batching = model_config_dict["dynamic_batching"]
            dynamic_batching_config = TritonDynamicBatchingConfig(
                preferred_batch_sizes=dynamic_batching.get("preferred_batch_size"),
                max_queue_delay_us=int(dynamic_batching.get("max_queue_delay_microseconds", 0)),
            )
        else:
            dynamic_batching_config = TritonDynamicBatchingConfig()

        capture_cuda_graphs = cls._get_tensorrt_capture_cuda_graphs(model_config_dict)
        optimization_config = TritonModelOptimizationConfig(
            **optimization_config_kwargs,
            tensorrt_capture_cuda_graph=(model_config_dict.get("platform") == "tensorrt_plan" and capture_cuda_graphs),
        )

        tensorrt_common_config = TensorRTCommonConfig(**tensorrt_common_config_kwargs)

        KIND_MAP = {
            "KIND_CPU": DeviceKind.CPU,
            "KIND_GPU": DeviceKind.GPU,
        }

        instances_config = TritonModelInstancesConfig(
            engine_count_per_device={
                KIND_MAP[entry["kind"]]: entry["count"] for entry in model_config_dict.get("instance_group", [])
            }
        )

        backend_parameters = model_config_dict.get("parameters", [])
        backend_parameters_config = TritonCustomBackendParametersConfig(
            triton_backend_parameters={name: backend_parameters[name]["string_value"] for name in backend_parameters}
        )

        explicit_format = None

        def _rewrite_io_spec(item):
            data_type = item["data_type"].split("_")[1]
            np_class = client_utils.triton_to_np_dtype(data_type)
            dummy_instance = np_class()
            dtype = dummy_instance.dtype if not isinstance(dummy_instance, bool) else np.dtype("bool")
            if len(item["dims"]) == 0:
                shape = (-1,)
            elif "reshape" in item:
                shape = (-1,) + tuple(map(lambda s: int(s), item["reshape"].get("shape", [])))
            else:
                shape = (-1,) + tuple(map(lambda s: int(s), item["dims"]))

            optional = item.get("optional", False)
            return TensorSpec(name=item["name"], shape=shape, dtype=dtype, optional=optional)

        inputs = {item["name"]: _rewrite_io_spec(item) for item in model_config_dict.get("input", [])} or None
        outputs = {item["name"]: _rewrite_io_spec(item) for item in model_config_dict.get("output", [])} or None
        signature = ModelSignatureConfig(inputs=inputs, outputs=outputs)
        model = Model(model_name, model_path, signature_if_missing=signature, explicit_format=explicit_format)

        return config_cls(
            model=model,
            batching_config=batching_config,
            triton_batching_config=triton_batching_config,
            optimization_config=optimization_config,
            dynamic_batching_config=dynamic_batching_config,
            tensorrt_common_config=tensorrt_common_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )

    @classmethod
    def _get_gpu_execution_accelerator(cls, model_config: Dict) -> List:
        if "optimization" not in model_config:
            return []

        optimization = model_config["optimization"]
        if "execution_accelerators" not in optimization:
            return []

        execution_accelerators = optimization["execution_accelerators"]
        if "gpu_execution_accelerator" not in execution_accelerators:
            return []

        return execution_accelerators["gpu_execution_accelerator"]

    @classmethod
    def _get_cpu_execution_accelerator(cls, model_config: Dict) -> List:
        if "optimization" not in model_config:
            return []

        optimization = model_config["optimization"]
        if "execution_accelerators" not in optimization:
            return []

        execution_accelerators = optimization["execution_accelerators"]
        if "cpu_execution_accelerator" not in execution_accelerators:
            return []

        return execution_accelerators["cpu_execution_accelerator"]

    @classmethod
    def _get_tensorrt_capture_cuda_graphs(cls, model_config: Dict) -> bool:
        if "optimization" not in model_config:
            return False

        optimization = model_config["optimization"]
        if "cuda" not in optimization:
            return False

        cuda = optimization["cuda"]
        if "graphs" not in cuda:
            return False

        return cuda["graphs"]


class TritonModelConfigGenerator:
    def __init__(
        self,
        model: Model,
        *,
        batching_config: BatchingConfig,
        triton_batching_config: TritonBatchingConfig,
        optimization_config: TritonModelOptimizationConfig,
        tensorrt_common_config: TensorRTCommonConfig,
        dynamic_batching_config: TritonDynamicBatchingConfig,
        instances_config: TritonModelInstancesConfig,
        backend_parameters_config: TritonCustomBackendParametersConfig,
        target_triton_version: Optional[str] = None,
    ):
        self._model = model
        self._batching_config = batching_config
        self._triton_batching_config = triton_batching_config
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
    def triton_batching_config(self):
        return self._triton_batching_config

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
        from tritonclient.grpc import model_config_pb2  # pytype: disable=import-error

        # https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
        model_config = {}
        backend_configurator = BackendConfiguratorSelector.for_model(self._model)
        backend_configurator.update_config_for_model(
            model_config,
            self._model,
            batching_config=self._batching_config,
            triton_batching_config=self._triton_batching_config,
            optimization_config=self._optimization_config,
            tensorrt_common_config=self._tensorrt_common_config,
            dynamic_batching_config=self._dynamic_batching_config,
            instances_config=self._instances_config,
            backend_parameters_config=self._backend_parameters_config,
        )

        LOGGER.debug(f"Generated Triton config:\n{json.dumps(model_config, indent=4)}")

        config_payload = json_format.ParseDict(model_config, model_config_pb2.ModelConfig())
        LOGGER.debug(f"Generated Triton config payload:\n{config_payload}")

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        model_config_bytes = text_format.MessageToBytes(config_payload)
        with config_path.open("wb") as cfg:
            cfg.write(model_config_bytes)

        return config_payload

    @classmethod
    def parse_triton_config_pbtxt(cls, config_path: Path, external_model_path: Optional[Path] = None):
        return ModelConfigParser.parse(config_path=config_path, external_model_path=external_model_path, config_cls=cls)
