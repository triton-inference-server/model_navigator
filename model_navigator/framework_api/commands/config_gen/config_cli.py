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

from pathlib import Path
from typing import Dict, Optional

from model_navigator.cli.convert_model import ConversionSetConfig
from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import SizedDataLoader
from model_navigator.framework_api.utils import (
    Extension,
    Framework,
    JitType,
    format_to_relative_model_path,
    get_package_path,
)
from model_navigator.model import Format, ModelConfig, ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.utils.config import YamlConfigFile


def extension2format(extension: str):
    if extension == Extension.ONNX:
        return Format.ONNX
    elif extension == Extension.SAVEDMODEL:
        return Format.TF_SAVEDMODEL
    elif extension == Extension.PT:
        return Format.TORCHSCRIPT
    elif extension == Extension.TRT:
        return Format.TENSORRT
    else:
        return None


class ConfigCli(Command):
    supported_extensions = [Extension.ONNX.value, Extension.PT.value, Extension.SAVEDMODEL.value]

    def __init__(
        self,
        target_format: Format,
        target_jit_type: Optional[JitType] = None,
        target_precision: Optional[TensorRTPrecision] = None,
    ):
        super().__init__(
            name="Generate configurations for Navigator CLI",
            command_type=CommandType.GEN_CONFIG,
            target_format=target_format,
        )
        self.target_jit_type = target_jit_type
        self.target_precision = target_precision

    def __call__(
        self,
        model,
        workdir: Path,
        framework: Framework,
        dataloader: SizedDataLoader,
        model_name: str,
        opset: int,
        input_metadata: Dict[str, TensorSpec],
        output_metadata: Dict[str, TensorSpec],
        **kwargs,
    ) -> Optional[Path]:
        model_path = format_to_relative_model_path(
            format=self.target_format, jit_type=self.target_jit_type, precision=self.target_precision
        )
        config_relative_path = model_path.parent / "config.yaml"
        config_path = get_package_path(workdir, model_name) / config_relative_path
        if config_path.is_file():
            return None
        with YamlConfigFile(config_path) as config_file:
            src_model_config = ModelConfig(
                model_name=model_name,
                model_path=model_path,
                model_format=self.target_format,
            )
            model_signature_config = ModelSignatureConfig(
                inputs=input_metadata,
                outputs=output_metadata,
            )
            conversion_set_config = ConversionSetConfig(
                onnx_opsets=[opset],
                target_formats=[self.target_format],
                tensorrt_precisions=[self.target_precision],
            )
            # comparator_config = ComparatorConfig()
            # dataset_profile_config = DatasetProfileConfig()

            config_file.save_config(src_model_config)
            config_file.save_config(model_signature_config)
            config_file.save_config(conversion_set_config)
            # config_file.save_config(comparator_config)
            # config_file.save_config(dataset_profile_config)

        return config_relative_path
