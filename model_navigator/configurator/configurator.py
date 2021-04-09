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
from pathlib import Path

from model_navigator.catalog import Catalog
from model_navigator.core import Format, Precision
from model_navigator.model import Model
from model_navigator.model_navigator_exceptions import ModelNavigatorException

from .model_configurator import (
    ONNXConfigurator,
    PyTorchConfigurator,
    TFConfigurator,
    TRTConfigurator,
)

_CONFIGURATORS = {
    Format.TF_SAVEDMODEL: TFConfigurator,
    Format.TRT: TRTConfigurator,
    Format.TS_SCRIPT: PyTorchConfigurator,
    Format.ONNX: ONNXConfigurator,
}


class Configurator:
    def get_models_variants(self, models: Catalog, max_batch_size: int):
        variants = Catalog()
        for model in models:
            model_name = model.name
            model_path = Path(model.path)
            if not model_path.is_file() and not model_path.is_dir():
                raise ModelNavigatorException(f"Model file not found in provided path: {model_path}")

            configurator_cls = self._get_type_configurator(model.format)
            configurator = configurator_cls()

            for variant in configurator.variants(model_name, max_batch_size):
                no_use_trt_on_triton_format_is_trt = variant.precision == Precision.ANY and model.format == Format.TRT
                precision = (
                    model.config.target_precisions[0] if no_use_trt_on_triton_format_is_trt else variant.precision
                )

                model_variant = Model(
                    path=model_path,
                    base_name=model.name,
                    name=variant.name,
                    format=variant.format,
                    max_batch_size=variant.max_batch_size,
                    precision=precision,
                    capture_cuda_graph=variant.capture_cuda_graph,
                    accelerator=variant.accelerator,
                    gpu_engine_count=variant.gpu_engine_count,
                    # here should be at least 1 element. More than 1 if not important
                    onnx_opset=model.config.onnx_opsets[0],
                )
                variants.add(model_variant)

        return variants

    def _get_type_configurator(self, model_format: Format):
        return _CONFIGURATORS[model_format]
