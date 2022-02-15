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
from model_navigator.model import Format, Model
from model_navigator.triton.backends.base import BaseBackendConfigurator
from model_navigator.triton.utils import rewrite_signature_to_model_config


class PyTorchBackendConfigurator(BaseBackendConfigurator):
    backend_name = "pytorch"
    supported_formats = [Format.TORCHSCRIPT]

    def _extract_signature(self, model_config, model: Model):
        rewrite_signature_to_model_config(model_config, model.signature)
