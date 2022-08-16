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
import dataclasses
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from model_navigator.constants import ALL_OTHER_INPUTS
from model_navigator.core import DEFAULT_TENSORRT_MAX_WORKSPACE_SIZE
from model_navigator.model import Format
from model_navigator.utils.config import BaseConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_TOLERANCE_VALUE = 1e-5


class ConversionLaunchMode(Enum):
    LOCAL = "local"
    DOCKER = "docker"


class TensorRTPrecision(Enum):
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


class TensorRTPrecisionMode(Enum):
    HIERARCHY = "hierarchy"
    SINGLE = "single"
    MIXED = "mixed"


TRITON_SUPPORTED_FORMATS = [
    Format.TF_TRT,
    Format.TF_SAVEDMODEL,
    Format.ONNX,
    Format.TENSORRT,
    Format.TORCHSCRIPT,
    Format.TORCH_TRT,
]


@dataclass
class ComparatorConfig(BaseConfig):
    """Key equal empty string is default tolerance value"""

    atol: Dict[str, float] = field(default_factory=lambda: {ALL_OTHER_INPUTS: DEFAULT_TOLERANCE_VALUE})
    rtol: Dict[str, float] = field(default_factory=lambda: {ALL_OTHER_INPUTS: DEFAULT_TOLERANCE_VALUE})


@dataclass
class DatasetProfileConfig(BaseConfig):
    min_shapes: Optional[Dict[str, Tuple]] = None
    opt_shapes: Optional[Dict[str, Tuple]] = None
    max_shapes: Optional[Dict[str, Tuple]] = None
    value_ranges: Optional[Dict[str, Tuple]] = None
    dtypes: Optional[Dict[str, np.dtype]] = None


@dataclass
class TensorRTConversionConfig(BaseConfig):
    precision: TensorRTPrecision = TensorRTPrecision.FP16
    precision_mode: TensorRTPrecisionMode = TensorRTPrecisionMode.HIERARCHY
    explicit_precision: bool = False
    sparse_weights: bool = False
    max_workspace_size: int = DEFAULT_TENSORRT_MAX_WORKSPACE_SIZE


@dataclass
class ConversionConfig(BaseConfig):
    target_format: Format

    # ONNX related
    onnx_opset: Optional[int] = None

    # TRT related
    tensorrt_config: TensorRTConversionConfig = field(default_factory=TensorRTConversionConfig)


class TargetFormatConfigSetIterator(ABC):
    def __init__(self, conversion_set_config, target_format=None):
        self._conversion_set_config = conversion_set_config
        self._target_format = target_format

    @abstractmethod
    def __iter__(self):
        pass

    @classmethod
    def for_target_format(cls, target_format, config_set):
        from model_navigator.converter.onnx.config import OnnxConfigSetIterator
        from model_navigator.converter.pyt.config import PyTorchConfigSetIterator
        from model_navigator.converter.tensorrt.config import TensorRTConfigSetIterator
        from model_navigator.converter.tf.config import TensorFlowConfigSetIterator
        from model_navigator.converter.tf_trt.config import TFTRTConfigSetIterator

        iterator_cls = {
            Format.ONNX: OnnxConfigSetIterator,
            Format.TENSORRT: TensorRTConfigSetIterator,
            Format.TF_SAVEDMODEL: TensorFlowConfigSetIterator,
            Format.TORCHSCRIPT: PyTorchConfigSetIterator,
            Format.TF_TRT: TFTRTConfigSetIterator,
            Format.TORCH_TRT: TensorRTConfigSetIterator,
        }[target_format]
        return iterator_cls(config_set, target_format=target_format)


@dataclasses.dataclass
class ConversionSetConfig(BaseConfig):
    target_formats: List[Format] = dataclasses.field(default_factory=lambda: TRITON_SUPPORTED_FORMATS)

    # ONNX related
    onnx_opsets: List[int] = dataclasses.field(default_factory=lambda: [14])

    # TRT related
    tensorrt_precisions: List[TensorRTPrecision] = dataclasses.field(
        default_factory=lambda: [TensorRTPrecision.FP16, TensorRTPrecision.FP32]
    )
    tensorrt_precisions_mode: TensorRTPrecisionMode = TensorRTPrecisionMode.HIERARCHY
    tensorrt_explicit_precision: bool = False
    tensorrt_sparse_weights: bool = False
    tensorrt_max_workspace_size: int = DEFAULT_TENSORRT_MAX_WORKSPACE_SIZE

    def __iter__(self):
        for target_format in self.target_formats:
            config_set_iterator = TargetFormatConfigSetIterator.for_target_format(target_format, self)
            yield from config_set_iterator

    @classmethod
    def from_single_config(cls, config: ConversionConfig):
        if not config.target_format:
            return cls(
                target_formats=[],
                tensorrt_precisions=[],
                onnx_opsets=[],
                tensorrt_precisions_mode=config.tensorrt_config.precision_mode,
                tensorrt_explicit_precision=config.tensorrt_config.explicit_precision,
                tensorrt_sparse_weights=config.tensorrt_config.sparse_weights,
                tensorrt_max_workspace_size=config.tensorrt_config.max_workspace_size,
            )

        return cls(
            target_formats=[config.target_format],
            onnx_opsets=[config.onnx_opset] if config.onnx_opset else [],
            tensorrt_precisions=[config.tensorrt_config.precision] or [],
            tensorrt_precisions_mode=config.tensorrt_config.precision_mode,
            tensorrt_explicit_precision=config.tensorrt_config.explicit_precision,
            tensorrt_sparse_weights=config.tensorrt_config.sparse_weights,
            tensorrt_max_workspace_size=config.tensorrt_config.max_workspace_size,
        )
