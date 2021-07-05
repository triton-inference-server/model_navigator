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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from model_navigator.exceptions import ModelNavigatorException
from model_navigator.model import ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.utils.config import YamlConfigFile
from model_navigator.utils.formats.base import BaseFormatUtils

LOGGER = logging.getLogger(__name__)


def validate_torchscript_signature(signature_config: ModelSignatureConfig):
    # based on
    # -  https://github.com/triton-inference-server/server/blob/89b7f8b30bf84d20f96825a6c476e7f71eca6dd6/docs/model_configuration.md#inputs-and-outputs

    SEP = "__"

    def _validate_io(io: Dict[str, TensorSpec], type_name: str):
        for name, spec in io.items():
            if name != spec.name:
                raise ModelNavigatorException(f"{type_name} name differs '{name}' != '{spec.name}'")
            if SEP not in name:
                raise ModelNavigatorException(
                    f"{type_name} '{name}' does not follow naming convention i.e. <name>__<index>."
                )
            index_str = name.split(SEP)[1]
            try:
                index = int(index_str)
                if index < 0:
                    raise ValueError()
            except ValueError:
                raise ModelNavigatorException(f"{type_name} '{name}' have invalid index value {index_str}.")
        io_indexes = [int(name.split(SEP)[1]) for name in io]
        if len(io) != len(set(io_indexes)):
            raise ModelNavigatorException(f"{type_name}s have duplicated indexes: {io_indexes}.")
        if io_indexes and max(io_indexes) - min(io_indexes) >= len(io):
            raise ModelNavigatorException(f"{type_name}s indexes are not subsequent: {io_indexes}.")

    inputs = signature_config.inputs or {}
    outputs = signature_config.outputs or {}
    _validate_io(inputs, "Input")
    _validate_io(outputs, "Output")


@dataclass
class TorchScriptProperties:
    pass


class TorchScriptUtils(BaseFormatUtils):
    @classmethod
    def get_signature(cls, path: Path):
        yaml_path = path.parent / f"{path.name}.yaml"

        try:
            with YamlConfigFile(yaml_path) as config_file:
                signature_config = config_file.load(ModelSignatureConfig)
                validate_torchscript_signature(signature_config)
                return signature_config
        except OSError:
            raise ValueError(f"TorchScript model configurator expects file {yaml_path} with tensor information.")

    @classmethod
    def get_properties(cls, path: Path):
        return TorchScriptProperties()
