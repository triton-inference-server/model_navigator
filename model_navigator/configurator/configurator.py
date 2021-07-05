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
from typing import Generator

from model_navigator.configurator.model_configurator import (
    ONNXConfigurator,
    PyTorchConfigurator,
    TFConfigurator,
    TRTConfigurator,
)
from model_navigator.configurator.variant import Variant
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.model import Format, Model

_CONFIGURATORS = {
    Format.TF_SAVEDMODEL: TFConfigurator,
    Format.TENSORRT: TRTConfigurator,
    Format.TORCHSCRIPT: PyTorchConfigurator,
    Format.ONNX: ONNXConfigurator,
}


class Configurator:
    def get_models_variants(self, model: Model) -> Generator[Variant, None, None]:
        if not model.path.is_file() and not model.path.is_dir():
            raise ModelNavigatorException(f"Model file not found in provided path: {model.path}")
        configurator_cls = self._get_type_configurator(model.format)
        configurator = configurator_cls()
        yield from configurator.variants(model)

    def _get_type_configurator(self, model_format: Format):
        return _CONFIGURATORS[model_format]
