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
from typing import Dict, List, Optional, Tuple

from model_navigator.converter.config import ComparatorConfig, ConversionSetConfig, TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.status import RuntimeResults
from model_navigator.framework_api.utils import (
    Extension,
    Framework,
    JitType,
    format_to_relative_model_path,
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
        name: str,
        target_format: Format,
        runtime_results: List[RuntimeResults],
        atol: Dict[str, float],
        rtol: Dict[str, float],
        requires: Tuple[Command, ...] = (),
        target_jit_type: Optional[JitType] = None,
        target_precision: Optional[TensorRTPrecision] = None,
        enable_xla: Optional[bool] = None,
        jit_compile: Optional[bool] = None,
    ):
        super().__init__(name=name, command_type=CommandType.GEN_CONFIG, requires=requires, target_format=target_format)
        self.target_jit_type = target_jit_type
        self.target_precision = target_precision
        self.runtime_results = runtime_results
        self.atol = atol
        self.rtol = rtol
        self.enable_xla = enable_xla
        self.jit_compile = jit_compile

    def __call__(
        self,
        workdir: Path,
        framework: Framework,
        model_name: str,
        opset: int,
        input_metadata: Dict[str, TensorSpec],
        output_metadata: Dict[str, TensorSpec],
        **kwargs,
    ) -> Optional[Path]:
        model_path = format_to_relative_model_path(
            format=self.target_format,
            jit_type=self.target_jit_type,
            precision=self.target_precision,
            enable_xla=self.enable_xla,
            jit_compile=self.jit_compile,
        )
        config_relative_path = model_path.parent / "config.yaml"
        config_path = workdir / config_relative_path
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

            comparator_config = ComparatorConfig(atol=self.atol, rtol=self.rtol)

            if opset:
                conversion_set_config = ConversionSetConfig()
                conversion_set_config.onnx_opsets = [opset]
                config_file.save_config(conversion_set_config, fields=["onnx_opsets"])

            config_file.save_config(src_model_config)
            config_file.save_config(model_signature_config)
            config_file.save_config(comparator_config)

        return config_relative_path
