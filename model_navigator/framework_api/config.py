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
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin

from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.framework_api.common import DataObject, SizedDataLoader, TensorMetadata
from model_navigator.framework_api.exceptions import UserError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import Framework, JitType, RuntimeProvider, pad_string
from model_navigator.model import Format


@dataclass
class Config(DataObject):
    # common params
    framework: Framework
    model_name: str
    model: object
    dataloader: SizedDataLoader
    workdir: Path
    override_workdir: bool
    target_formats: Tuple[Format, ...]
    sample_count: int
    disable_git_info: bool
    batch_dim: Optional[int] = 0
    seed: int = 0
    timestamp: Optional[str] = None
    _input_names: Optional[Tuple[str, ...]] = None
    _output_names: Optional[Tuple[str, ...]] = None
    from_source: bool = True
    max_batch_size: Optional[int] = None
    input_metadata: Optional[TensorMetadata] = None
    output_metadata: Optional[TensorMetadata] = None
    profiler_config: Optional[ProfilerConfig] = None
    forward_kw_names: Optional[Tuple[str, ...]] = None

    # TRT params
    max_workspace_size: Optional[int] = None
    target_precisions: Optional[Tuple[TensorRTPrecision, ...]] = None
    precision_mode: Optional[TensorRTPrecisionMode] = None
    trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None

    # TF-TRT params
    minimum_segment_size: Optional[int] = None

    # PyTorch
    target_jit_type: Optional[Tuple[JitType, ...]] = None
    target_device: str = "cpu"

    # ONNX
    opset: Optional[int] = None
    dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None
    runtimes: Tuple[RuntimeProvider, ...] = ()

    # JAX
    model_params: Optional[Any] = None
    jit_compile: Optional[Tuple[bool]] = None
    enable_xla: Optional[Tuple[bool]] = None

    # Correctness is computed using allclose function for all tensors
    # for output from converted model and source model
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    atol: Optional[float] = None
    rtol: Optional[float] = None

    def _check_types(self):
        try:
            iter(self.dataloader)
        except TypeError as e:
            raise TypeError("Datalaoder must be iterable.") from e
        try:
            len(self.dataloader)
        except TypeError as e:
            raise TypeError("Datalaoder must must have len().") from e

        for field_name, field in self.__dataclass_fields__.items():
            expected_type = field.type
            if get_origin(expected_type) is Union:
                expected_type = tuple((get_origin(arg) or arg) for arg in get_args(expected_type))
            else:
                expected_type = get_origin(expected_type) or expected_type
            value = getattr(self, field_name)

            if isinstance(expected_type, (list, tuple)) and Any in expected_type:
                continue

            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Incorrect type for {field_name}. Expected type {expected_type} got {type(value)} instead."
                )

        if isinstance(self.workdir, str):
            object.__setattr__(self, "workdir", Path(self.workdir))

    def _log(self):
        LOGGER.debug(pad_string("Config parameters"))
        log_dict = self.to_dict(
            filter_fields=[
                "model",
                "dataloader",
                "input_metadata",
                "output_metadata",
                "forward_kw_names",
            ],
            parse=True,
        )
        LOGGER.debug(log_dict)

    def __post_init__(self):
        self._check_types()
        if "/" in self.model_name:
            raise UserError("Model name cannot contain '/' character.")  # '/' causes problems for OTIS
        object.__setattr__(self, "timestamp", f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S.%f}")
        self._log()
