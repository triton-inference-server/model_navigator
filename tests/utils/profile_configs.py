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
import pathlib

from google.protobuf import json_format, text_format  # pytype: disable=pyi-error
from tritonclient.grpc import model_config_pb2  # pytype: disable=pyi-error

from model_navigator.exceptions import ModelNavigatorException
from model_navigator.utils.devices import get_available_gpus


class ModelConfig:
    def __init__(self, model_config):
        self._model_config = model_config

    @staticmethod
    def create_from_file(model_path: pathlib.Path):
        if not model_path.exists():
            raise ModelNavigatorException(f'Model path "{model_path}" specified does not exist.')

        if not model_path.is_dir():
            raise ModelNavigatorException(f'Model output path "{model_path}" must be a directory.')

        model_config_path = model_path / "config.pbtxt"
        if not model_config_path.is_file():
            raise ModelNavigatorException(
                f'Path "{model_config_path}" does not exist.'
                " Make sure that you have specified the correct model"
                " repository and model name(s)."
            )

        with model_config_path.open("r+") as f:
            config_str = f.read()

        protobuf_message = text_format.Parse(config_str, model_config_pb2.ModelConfig())

        return ModelConfig(protobuf_message)

    def to_dict(self):
        model_config_dict = json_format.MessageToDict(self._model_config)
        model_config_dict["cpu_only"] = self._cpu_only(model_config_dict)
        return model_config_dict

    def _cpu_only(self, model_config_dict):
        cpu_only = True
        if "instanceGroup" in model_config_dict:
            instance_group_list = model_config_dict["instanceGroup"]
            for group in instance_group_list:
                if "kind" in group and group["kind"] != "KIND_CPU":
                    cpu_only = False
        else:
            cpu_only = False

        return cpu_only


def get_profiling_configs(workspace):
    model_repository = workspace.path / "analyzer" / "model-store"

    configs = {}
    for model_config in sorted(model_repository.iterdir(), key=lambda item: item.name):
        config_data = ModelConfig.create_from_file(model_config).to_dict()
        configs[model_config.name] = config_data

    return configs


def parse_expected_configs_jsonlines(expected_configs_jsonlines: str):

    gpu_available = bool(get_available_gpus())
    markers_mapping = {
        "__KIND_HW_MARKER__": "KIND_GPU" if gpu_available else "KIND_CPU",
        "__CPU_ONLY_MARKER__": "false" if gpu_available else "true",
    }

    for marker, fill in markers_mapping.items():
        expected_configs_jsonlines = expected_configs_jsonlines.replace(marker, fill)
    return expected_configs_jsonlines
