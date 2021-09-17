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
from typing import Optional

from model_navigator.exceptions import ModelNavigatorDeployerException
from model_navigator.model import Model
from model_navigator.triton import (
    DeviceKind,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
    TritonModelSchedulerConfig,
)
from model_navigator.triton.client import grpc_client
from model_navigator.triton.config import TritonCustomBackendParametersConfig
from model_navigator.utils.formats import FORMAT2SUFFIX

LOGGER = logging.getLogger(__name__)

ModelConfigProtobufType = grpc_client.model_config_pb2.ModelConfig


class BaseBackendConfigurator:
    backend_name = None
    platform_name = None
    supported_formats = None

    def __init__(self):
        self._target_triton_version = None

    def update_config_for_model(
        self,
        model_config: ModelConfigProtobufType,
        model: Model,
        *,
        optimization_config: Optional[TritonModelOptimizationConfig] = None,
        scheduler_config: Optional[TritonModelSchedulerConfig] = None,
        instances_config: Optional[TritonModelInstancesConfig] = None,
        backend_parameters_config: Optional[TritonCustomBackendParametersConfig] = None,
    ):
        model_config.name = model.name
        if self.backend_name is not None:
            model_config.backend = self.backend_name
        if self.platform_name is not None:
            model_config.platform = self.platform_name
        self._extract_signature(model_config, model)
        self._set_backend_acceleration(model_config, optimization_config or TritonModelOptimizationConfig())
        self._set_scheduler(model_config, scheduler_config=scheduler_config or TritonModelSchedulerConfig())
        self._set_instance_group(model_config, instances_config=instances_config or TritonModelInstancesConfig())
        self._set_custom_backend_parameters(
            model_config, backend_parameters_config or TritonCustomBackendParametersConfig()
        )
        return model_config

    def _extract_signature(self, model_config, model: Model):
        pass

    def _set_backend_acceleration(self, model_config, optimization_config: TritonModelOptimizationConfig):
        pass

    def _set_scheduler(self, model_config, scheduler_config: TritonModelSchedulerConfig):
        model_config.max_batch_size = scheduler_config.max_batch_size
        if any([scheduler_config.preferred_batch_sizes, scheduler_config.max_queue_delay_us > 0]):
            model_support_batching = scheduler_config.max_batch_size > 0
            if model_support_batching:
                model_config.dynamic_batching.max_queue_delay_microseconds = max(
                    int(scheduler_config.max_queue_delay_us), 0
                )
                preferred_batch_sizes = scheduler_config.preferred_batch_sizes or [scheduler_config.max_batch_size]
                for preferred_batch_size in preferred_batch_sizes:
                    model_config.dynamic_batching.preferred_batch_size.append(int(preferred_batch_size))
            else:
                LOGGER.warning("Ignore dynamic batching parameters as model doesn't support batching")

    def _set_instance_group(self, model_config, instances_config: TritonModelInstancesConfig):
        for kind, count in instances_config.engine_count_per_device.items():
            instance_group = model_config.instance_group.add()
            instance_group.kind = {
                DeviceKind.CPU: instance_group.KIND_CPU,
                DeviceKind.GPU: instance_group.KIND_GPU,
            }[kind]
            instance_group.count = count

    def _set_custom_backend_parameters(
        self, model_config, backend_parameters_config: TritonCustomBackendParametersConfig
    ):
        for key, value in backend_parameters_config.triton_backend_parameters.items():
            model_config.parameters[key].string_value = str(value)

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
