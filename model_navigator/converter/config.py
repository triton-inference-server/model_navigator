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
    FP16 = "fp16"
    FP32 = "fp32"
    TF32 = "tf32"


@dataclass
class ComparatorConfig(BaseConfig):
    """Key equal empty string is default tolerance value"""

    atol: Dict[str, float] = field(default_factory=lambda: {ALL_OTHER_INPUTS: DEFAULT_TOLERANCE_VALUE})
    rtol: Dict[str, float] = field(default_factory=lambda: {ALL_OTHER_INPUTS: DEFAULT_TOLERANCE_VALUE})
    max_batch_size: int = 32


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
    target_precision: Optional[TensorRTPrecision] = None
    # ONNX related
    onnx_opset: Optional[int] = None
    # TRT related
    max_workspace_size: Optional[int] = None
