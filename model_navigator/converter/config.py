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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from model_navigator.constants import ALL_OTHER_INPUTS
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
    TF32 = "tf32"


class TensorRTPrecisionMode(Enum):
    HIERARCHY = "hierarchy"
    SINGLE = "single"
    MIXED = "mixed"


@dataclass
class ComparatorConfig(BaseConfig):
    """Key equal empty string is default tolerance value"""

    atol: Dict[str, float] = field(default_factory=lambda: {ALL_OTHER_INPUTS: DEFAULT_TOLERANCE_VALUE})
    rtol: Dict[str, float] = field(default_factory=lambda: {ALL_OTHER_INPUTS: DEFAULT_TOLERANCE_VALUE})
    max_batch_size: int = 1


@dataclass
class DatasetProfileConfig(BaseConfig):
    min_shapes: Optional[Dict[str, Tuple]] = None
    opt_shapes: Optional[Dict[str, Tuple]] = None
    max_shapes: Optional[Dict[str, Tuple]] = None
    value_ranges: Optional[Dict[str, Tuple]] = None
    dtypes: Optional[Dict[str, np.dtype]] = None


@dataclass
class ConversionConfig(BaseConfig):
    target_format: Format

    # ONNX related
    onnx_opset: Optional[int] = None

    # TRT related
    tensorrt_precision: Optional[TensorRTPrecision] = None
    tensorrt_precision_mode: TensorRTPrecisionMode = TensorRTPrecisionMode.HIERARCHY
    tensorrt_explicit_precision: bool = False
    tensorrt_strict_types: bool = False
    tensorrt_sparse_weights: bool = False


class TargetFormatConfigSetIterator(ABC):
    def __init__(self, conversion_set_config):
        self._conversion_set_config = conversion_set_config

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
        }[target_format]
        return iterator_cls(config_set)
