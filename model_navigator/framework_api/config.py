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

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.common import SizedDataLoader
from model_navigator.framework_api.utils import DataObject, Framework, JitType
from model_navigator.model import Format


@dataclass(frozen=True)
class Config(DataObject):
    # common params
    framework: Framework
    model_name: str
    model: object
    dataloader: SizedDataLoader
    workdir: Path
    override_workdir: bool
    keep_workdir: bool
    target_formats: Tuple[Format]
    sample_count: int
    save_data: bool
    disable_git_info: bool
    batch_dim: Optional[int] = 0
    seed: int = 0
    timestamp: Optional[str] = None
    _input_names: Optional[Tuple[str]] = None
    _output_names: Optional[Tuple[str]] = None

    # TRT params
    max_workspace_size: Optional[int] = None
    target_precisions: Optional[Tuple[TensorRTPrecision]] = None
    trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None

    # TF-TRT params
    minimum_segment_size: Optional[int] = None

    # PyTorch
    target_jit_type: Optional[Tuple[JitType]] = None
    forward_kw_names: Optional[Tuple[str]] = None
    target_device: str = "cpu"

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
