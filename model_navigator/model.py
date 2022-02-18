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
from dataclasses import InitVar, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from model_navigator.exceptions import ModelNavigatorException
from model_navigator.tensor import TensorSpec
from model_navigator.utils.config import BaseConfig


class Format(Enum):
    TORCHSCRIPT = "torchscript"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TORCH_TRT = "torch-trt"
    ONNX = "onnx"
    TENSORRT = "trt"


@dataclass
class ModelConfig(BaseConfig):
    model_name: str
    model_path: Path
    model_format: Optional[Format] = None
    model_version: str = "1"


@dataclass
class ModelSignatureConfig(BaseConfig):
    inputs: Optional[Dict[str, TensorSpec]] = None
    outputs: Optional[Dict[str, TensorSpec]] = None

    def has_input_dynamic_axes(self) -> Optional[bool]:
        # FIXME: for signatures of models which doesn't support batching
        has_dynamic_axes = None
        if self.inputs:
            has_dynamic_axes = any([spec.is_dynamic() for name, spec in self.inputs.items()])
        return has_dynamic_axes

    def is_missing(self):
        return self.inputs is None or self.outputs is None


@dataclass
class Model(BaseConfig):
    name: str
    path: Path
    format: Format = field(init=False)
    signature: Optional[ModelSignatureConfig] = field(init=False)
    properties: Any = field(init=False)
    num_required_gpus: Optional[int] = field(init=False)
    signature_if_missing: InitVar[Optional[ModelSignatureConfig]] = None
    explicit_format: InitVar[Optional[Format]] = None

    def __post_init__(self, signature_if_missing: Optional[ModelSignatureConfig], explicit_format: Optional[Format]):
        from model_navigator.utils.formats import FORMAT2ADAPTER

        self.format = self._get_format(explicit_format)

        adapter = FORMAT2ADAPTER[self.format]
        self.signature = self._get_signature(adapter, signature_if_missing)
        self.properties = adapter.get_properties(self.path)
        self.num_required_gpus = adapter.get_num_required_gpus(self.properties)

    def _get_signature(self, adapter, signature_if_missing):
        signature = adapter.get_signature(self.path)
        if (
            signature is not None
            and signature.is_missing()
            and signature_if_missing
            and not signature_if_missing.is_missing()
        ):
            adapter.validate_signature(signature_if_missing)
            signature = signature_if_missing
        return signature

    def _get_format(self, explicit_format):
        from .utils.formats import SUFFIX2FORMAT, guess_format

        model_format = explicit_format
        if not model_format:
            model_format = guess_format(self.path)

        if not model_format:
            raise ModelNavigatorException(
                f"""Unsupported file type in {self.path}. """
                """Please provide file with one of the following file type: """
                f"""{", ".join(list(map(lambda ext: f"*{ext}", SUFFIX2FORMAT.keys())))} """
                """or use --model-format parameter"""
            )

        return model_format
