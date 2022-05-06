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
import copy

from model_navigator.model import Format, Model
from model_navigator.triton.backends.base import BaseBackendConfigurator
from model_navigator.triton.utils import rewrite_signature_to_model_config


class PyTorchBackendConfigurator(BaseBackendConfigurator):
    backend_name = "pytorch"
    supported_formats = [
        Format.TORCHSCRIPT,
        Format.TORCH_TRT,
    ]

    def _extract_signature(self, model_config, model: Model):
        tmp_sig = copy.copy(model.signature)

        # rewrite outputs to use the triton convention, see:
        # https://github.com/triton-inference-server/server/blob/89b7f8b30bf84d20f96825a6c476e7f71eca6dd6/docs/model_configuration.md#inputs-and-outputs
        # FIXME: This assumes that outputs are given in proper order
        new_outputs = {}
        for i, out in enumerate(tmp_sig.outputs):
            name = f"output__{i}"
            spec_ = copy.copy(tmp_sig.outputs[out])
            spec_.name = name
            new_outputs[name] = spec_
        tmp_sig.outputs = new_outputs

        rewrite_signature_to_model_config(model_config, tmp_sig)
