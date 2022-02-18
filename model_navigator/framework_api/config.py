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

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.utils import DataObject, Framework, JitType
from model_navigator.model import Format
from model_navigator.tensor import TensorSpec


@dataclass(frozen=True)
class Config(DataObject):
    # common params
    framework: Framework
    model_name: str
    model: object
    dataloader: Callable
    workdir: Path
    override_workdir: bool
    keep_workdir: bool
    target_formats: Tuple[Format]
    sample_count: int
    input_metadata: Dict[str, TensorSpec]
    output_metadata: Dict[str, TensorSpec]
    save_data: bool
    timestamp: Optional[str] = None

    # TF-TRT params
    max_workspace_size: Optional[int] = None
    target_precisions: Optional[Tuple[TensorRTPrecision]] = None
    minimum_segment_size: Optional[int] = None

    # PyTorch
    target_jit_type: Optional[Tuple[JitType]] = None
    forward_kw_names: Optional[Tuple[str]] = None

    # ONNX
    opset: Optional[int] = None
    dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None

    # Correctness is computed using allclose function for all tensors
    # for output from converted model and source model
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    atol: Optional[float] = None
    rtol: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "timestamp", f"{datetime.datetime.now():%Y-%m-%dT%H:%M:%S.%f}")
        object.__setattr__(self, "input_names", tuple(self.input_metadata.keys()))
        object.__setattr__(self, "output_names", tuple(self.output_metadata.keys()))
