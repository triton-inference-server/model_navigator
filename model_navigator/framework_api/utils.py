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
import pathlib
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import numpy

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.common import Sample
from model_navigator.model import Format


def get_available_onnx_providers() -> List:
    import onnxruntime as onnxrt  # pytype: disable=import-error

    return onnxrt.get_available_providers()


def numpy_to_torch_dtype(np_dtype):
    np_dtype = numpy.dtype(np_dtype).type
    import torch  # pytype: disable=import-error

    return {
        numpy.bool_: torch.bool,
        numpy.uint8: torch.uint8,
        numpy.int8: torch.int8,
        numpy.int16: torch.int16,
        numpy.int32: torch.int32,
        numpy.int64: torch.int64,
        numpy.float16: torch.float16,
        numpy.float32: torch.float32,
        numpy.float64: torch.float64,
        numpy.complex64: torch.complex64,
        numpy.complex128: torch.complex128,
    }[np_dtype]


class Parameter(Enum):
    def __str__(self):
        return self.value

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(f"{cls.name}: {node._value_}")


class RuntimeProvider(str, Parameter):
    TRT = "TensorrtExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    CPU = "CPUExecutionProvider"
    TF = "TensorFlowExecutionProvider"
    PYT = "PyTorchExecutionProvider"


def format2runtimes(format: Format) -> Optional[Tuple]:
    if format == format.ONNX:
        return parse_enum(get_available_onnx_providers(), RuntimeProvider)
    elif format == format.TORCHSCRIPT or format == format.TORCH_TRT:
        return (RuntimeProvider.PYT,)
    elif format == format.TF_SAVEDMODEL or format == format.TF_TRT:
        return (RuntimeProvider.TF,)
    elif format == format.TENSORRT:
        return (RuntimeProvider.TRT,)
    else:
        return None


class Status(str, Parameter):
    OK = "OK"
    FAIL = "FAIL"
    NOOP = "NOOP"
    INITIALIZED = "INITIALIZED"
    SKIPPED = "SKIPPED"


class Framework(Parameter):
    TF2 = "tensorflow2"
    PYT = "pytorch"


class Indent(str, Parameter):
    SINGLE = "  "
    DOUBLE = 2 * SINGLE
    TRIPLE = 3 * SINGLE


class Extension(Parameter):
    ONNX = "onnx"
    PT = "pt"
    SAVEDMODEL = "savedmodel"
    TRT = "plan"


class JitType(Parameter):
    SCRIPT = "script"
    TRACE = "trace"


class ArtifactType(Parameter):
    EXPORTED_MODEL_PATH = "exported_model_path"
    CONVERTED_MODEL_PATH = "converted_model_path"
    NAVIGATOR_CLI_CONFIG_PATH = "navigator_cli_config_path"
    NAVIGATOR_CONFIG_PATH = "navigator_config_path"


def get_default_model_name():
    return "navigator_model"


def get_default_workdir():
    return Path.cwd() / "navigator_workdir"


def get_default_max_workspace_size():
    return 8589934592


def pad_string(s: str):
    s = f"{30 * '='} {s} "
    s = s.ljust(100, "=")
    return s


def to_numpy(tensor, from_framework: Framework):
    if isinstance(tensor, numpy.ndarray):
        return tensor
    if from_framework == Framework.PYT:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.numpy()


def get_package_path(workdir: Path, model_name: str):
    return workdir / f"{model_name}.nav"


# pytype: disable=wrong-arg-types
def format_to_relative_model_path(
    format: Format, jit_type: JitType = JitType.SCRIPT, precision: TensorRTPrecision = TensorRTPrecision.FP32
):
    if format == Format.ONNX:
        return Path(f"{format.value}") / "model.onnx"
    if format == Format.TORCHSCRIPT:
        return Path(f"{format.value}-{jit_type.value}") / "model.pt"
    if format == Format.TORCH_TRT:
        return Path(f"{format.value}-{jit_type.value}") / "model.pt"
    if format == Format.TF_SAVEDMODEL:
        return Path(format.value) / "model.savedmodel"
    if format == Format.TF_TRT:
        return Path(f"{format.value}-{precision.value}") / "model.savedmodel"
    if format == Format.TENSORRT:
        return Path(f"{format.value}-{precision.value}") / "model.plan"
    else:
        return Path(f"unknown-format-{format}")


# pytype: enable=wrong-arg-types


class DataObject:
    def to_dict(self, filter_fields: Optional[List[str]] = None, parse: bool = False):
        data = {}

        if filter_fields:
            filtered_data = {key: value for key, value in self.__dict__.items() if key not in filter_fields}
        else:
            filtered_data = self.__dict__

        if parse:
            for key, value in filtered_data.items():
                if value is None:
                    continue
                data[key] = self._parse_value(value)
        else:
            data = filtered_data

        return data

    def _parse_value(self, value):
        if isinstance(value, DataObject):
            value = value.to_dict(parse=True)
        elif isinstance(value, Mapping):
            value = self._from_dict(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            value = self._from_list(value)
        elif isinstance(value, Enum):
            value = value.value
        elif isinstance(value, pathlib.Path):
            value = str(value)

        return value

    def _from_dict(self, values):
        data = {}
        for key, value in values.items():
            data[key] = self._parse_value(value)

        return data

    def _from_list(self, values):
        items = []
        for value in values:
            item = self._parse_value(value)
            items.append(item)

        return items


def sample_to_tuple(input: Any) -> Tuple[Any, ...]:
    if isinstance(input, tuple):
        return input
    if isinstance(input, list):
        return tuple(input)
    if isinstance(input, Mapping):
        return tuple(input.values())
    return (input,)


def extract_bs1(sample: Sample, batch_dim: Optional[int]) -> Sample:
    if batch_dim is not None:
        return {name: tensor.take([0], batch_dim) for name, tensor in sample.items()}
    return sample


def parse_enum(value, enum_type):
    if value is not None:
        value = tuple(value) if isinstance(value, (tuple, list)) else (value,)
        value = tuple(enum_type(v) for v in value)
    return value
