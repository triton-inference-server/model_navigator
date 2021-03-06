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
from typing import Dict, Optional

from model_navigator.common.config import BatchingConfig, TensorRTCommonConfig
from model_navigator.exceptions import ModelNavigatorDeployerException
from model_navigator.model import Model
from model_navigator.triton import (
    DeviceKind,
    TritonDynamicBatchingConfig,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
)
from model_navigator.triton.client import grpc_client
from model_navigator.triton.config import Batching, TritonBatchingConfig, TritonCustomBackendParametersConfig
from model_navigator.triton.utils import rewrite_signature_to_model_config
from model_navigator.utils.formats import FORMAT2SUFFIX

LOGGER = logging.getLogger(__name__)

if grpc_client:
    ModelConfigProtobufType = grpc_client.model_config_pb2.ModelConfig  # pytype: disable=attribute-error
else:
    ModelConfigProtobufType = object


class BaseBackendConfigurator:
    backend_name = None
    platform_name = None
    supported_formats = None

    def __init__(self):
        self._target_triton_version = None

    # pytype: disable=invalid-annotation
    def update_config_for_model(
        self,
        model_config: Dict,
        model: Model,
        batching_config: BatchingConfig,
        triton_batching_config: TritonBatchingConfig,
        *,
        optimization_config: Optional[TritonModelOptimizationConfig] = None,
        dynamic_batching_config: Optional[TritonDynamicBatchingConfig] = None,
        tensorrt_common_config: Optional[TensorRTCommonConfig] = None,
        instances_config: Optional[TritonModelInstancesConfig] = None,
        backend_parameters_config: Optional[TritonCustomBackendParametersConfig] = None,
    ):
        model_config["name"] = model.name
        if self.backend_name is not None:
            model_config["backend"] = self.backend_name
        if self.platform_name is not None:
            model_config["platform"] = self.platform_name
        self._extract_signature(model_config, model)
        self._set_batching(
            model_config,
            batching_config=batching_config,
            triton_batching_config=triton_batching_config,
            dynamic_batching_config=dynamic_batching_config or TritonDynamicBatchingConfig(),
        )
        self._set_backend_acceleration(
            model_config,
            optimization_config or TritonModelOptimizationConfig(),
            tensorrt_common_config or TensorRTCommonConfig(),
        )
        self._set_instance_group(model_config, instances_config=instances_config or TritonModelInstancesConfig())
        self._set_custom_backend_parameters(
            model_config, backend_parameters_config or TritonCustomBackendParametersConfig()
        )
        return model_config

    # pytype: enable=invalid-annotation

    def _extract_signature(self, model_config, model: Model):
        if model.signature and not model.signature.is_missing():
            rewrite_signature_to_model_config(model_config, model.signature)

    def _set_backend_acceleration(
        self,
        model_config,
        optimization_config: TritonModelOptimizationConfig,
        tensorrt_common_config: TensorRTCommonConfig,
    ):
        pass

    def _set_batching(
        self,
        model_config,
        batching_config: BatchingConfig,
        triton_batching_config: TritonBatchingConfig,
        dynamic_batching_config: TritonDynamicBatchingConfig,
    ):
        if triton_batching_config.batching == Batching.DISABLED:
            model_config["max_batch_size"] = 0
            LOGGER.debug("Batching for model is disabled. Supported request batch size=1.")
            return

        model_config["max_batch_size"] = batching_config.max_batch_size
        if triton_batching_config.batching == Batching.DYNAMIC:
            dynamic_batching = {}
            if dynamic_batching_config.max_queue_delay_us > 0:
                dynamic_batching["maxQueueDelayMicroseconds"] = int(dynamic_batching_config.max_queue_delay_us)

            if dynamic_batching_config.preferred_batch_sizes:
                dynamic_batching["preferredBatchSize"] = [
                    int(preferred_batch_size) for preferred_batch_size in dynamic_batching_config.preferred_batch_sizes
                ]
            model_config["dynamic_batching"] = dynamic_batching
        else:
            LOGGER.debug("Default batching used")

    def _set_instance_group(self, model_config, instances_config: TritonModelInstancesConfig):
        instance_groups = []
        for kind, count in instances_config.engine_count_per_device.items():
            config_kind = {
                DeviceKind.CPU: "KIND_CPU",
                DeviceKind.GPU: "KIND_GPU",
            }[kind]
            instance_groups.append(
                {
                    "count": count,
                    "kind": config_kind,
                }
            )

        if instance_groups:
            model_config["instance_group"] = instance_groups

    def _set_custom_backend_parameters(
        self, model_config, backend_parameters_config: TritonCustomBackendParametersConfig
    ):
        parameters = {}
        for key, value in backend_parameters_config.triton_backend_parameters.items():
            parameters[key] = {
                "string_value": str(value),
            }

        if parameters:
            model_config["parameters"] = parameters

    @classmethod
    def is_supporting_model(cls, model):
        return cls.supported_formats and model.format in cls.supported_formats

    def get_filename(self, model: Model):
        suffix = FORMAT2SUFFIX[model.format]
        return f"model{suffix}"


class BackendConfiguratorSelector:
    @classmethod
    def for_model(cls, model: Model, **kwargs):
        # add here new backends to be registered as subclass
        from .onnx import OnnxBackendConfigurator  # noqa: F401
        from .pytorch import PyTorchBackendConfigurator  # noqa: F401
        from .tensorflow import TensorFlowBackendConfigurator  # noqa: F401
        from .tensorrt import TensorRTBackendConfigurator  # noqa: F401

        cls_list = BaseBackendConfigurator.__subclasses__()
        cls_supporting_model = [cls_ for cls_ in cls_list if cls_.is_supporting_model(model)]
        if len(cls_supporting_model) > 1:
            raise ModelNavigatorDeployerException(
                f"More than single backend configurator supports model {model}: {cls_supporting_model}"
            )
        elif not cls_supporting_model:
            raise ModelNavigatorDeployerException(f"Missing backend configurators supporting model {model}")
        cls_supporting_model = cls_supporting_model[0]
        return cls_supporting_model(**kwargs)
