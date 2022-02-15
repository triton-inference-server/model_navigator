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

from model_navigator.model import ModelSignatureConfig
from model_navigator.utils.config import YamlConfigFile


def load_annotation(input_model_path: pathlib.Path) -> ModelSignatureConfig:
    annotation_path = input_model_path.parent / f"{input_model_path.name}.yaml"
    with YamlConfigFile(annotation_path) as config_file:
        io_spec = config_file.load(ModelSignatureConfig)  # pytype: disable=annotation-type-mismatch
    return io_spec
