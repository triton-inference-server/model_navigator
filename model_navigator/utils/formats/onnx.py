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
import logging
from dataclasses import dataclass
from pathlib import Path

from model_navigator.model import ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.utils.formats.base import BaseFormatUtils

LOGGER = logging.getLogger(__name__)


@dataclass
class ONNXProperties:
    onnx_opset: int


class ONNXUtils(BaseFormatUtils):
    @classmethod
    def get_signature(cls, path: Path):
        import onnx
        from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata

        model = onnx.load(path.as_posix())

        input_metadata = get_input_metadata(model.graph)
        output_metadata = get_output_metadata(model.graph)

        return ModelSignatureConfig(
            inputs={
                name: TensorSpec.from_polygraphy_metadata_tuple(name, metadata_tuple)
                for name, metadata_tuple in input_metadata.items()
            },
            outputs={
                name: TensorSpec.from_polygraphy_metadata_tuple(name, metadata_tuple)
                for name, metadata_tuple in output_metadata.items()
            },
        )

    @classmethod
    def validate_signature(cls, signature: ModelSignatureConfig):
        pass

    @classmethod
    def get_properties(cls, path: Path):
        import onnx

        model = onnx.load(path.as_posix())
        try:
            opset = model.opset_import[0].version
        except Exception:
            LOGGER.warning("Model does not contain ONNX opset information")
            opset = None
        return ONNXProperties(onnx_opset=opset)
