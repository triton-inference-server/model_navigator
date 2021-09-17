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
import itertools
from enum import Enum
from typing import Generator

from model_navigator.configurator.variant import Variant
from model_navigator.converter import PARAMETERS_SEP
from model_navigator.model import Model
from model_navigator.triton import TritonModelOptimizationConfig
from model_navigator.triton.config import BackendAccelerator, TensorRTOptPrecision

MODEL_CONFIG_SEP = "."


class ModelConfigurator:
    accelerators = (None,)
    capture_cuda_graph = (None,)

    def variants(self, model: Model) -> Generator[Variant, None, None]:
        parameters = [self.accelerators, self.capture_cuda_graph]
        combinations = itertools.product(*parameters)
        for combination in combinations:
            accelerator, capture_cuda_graph = combination
            if accelerator == BackendAccelerator.TRT:
                precisions = (TensorRTOptPrecision.FP16, TensorRTOptPrecision.FP32)
            else:
                precisions = (None,)

            for precision in precisions:
                optimization_config = TritonModelOptimizationConfig(
                    backend_accelerator=accelerator,
                    tensorrt_precision=precision,
                    tensorrt_capture_cuda_graph=bool(capture_cuda_graph or False),
                )

                variant_name = self._get_variant_name(model.name, [accelerator, capture_cuda_graph, precision])
                yield Variant(variant_name, optimization_config)

    def _get_variant_name(self, model_name, parameters):
        def _format_param(param):
            return str(param.value if isinstance(param, Enum) else param)

        variant_names = [_format_param(parameter) for parameter in parameters if parameter is not None]
        if not variant_names:
            return model_name

        suffix = PARAMETERS_SEP.join(variant_names)
        return f"{model_name}{MODEL_CONFIG_SEP}{suffix}"


class TFConfigurator(ModelConfigurator):
    accelerators = (
        None,
        BackendAccelerator.AMP,
        BackendAccelerator.TRT,
    )


class PyTorchConfigurator(ModelConfigurator):
    accelerators = (None,)


class ONNXConfigurator(ModelConfigurator):
    accelerators = (
        None,
        BackendAccelerator.TRT,
    )


class TRTConfigurator(ModelConfigurator):
    capture_cuda_graph = (0, 1)
    accelerators = (None,)
