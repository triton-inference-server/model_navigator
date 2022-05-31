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
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import numpy

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.common import Sample
from model_navigator.framework_api.exceptions import UserError
from model_navigator.model import Format
from model_navigator.utils.device import get_available_gpus

T = TypeVar("T")


def get_available_onnx_providers() -> List:
    import onnxruntime as onnxrt  # pytype: disable=import-error

    onnx_providers = onnxrt.get_available_providers()
    if not get_available_gpus():  # filter out providers that require GPU
        gpu_providers = [RuntimeProvider.CUDA, RuntimeProvider.TRT]
        onnx_providers = [prov for prov in onnx_providers if prov not in gpu_providers]
    return onnx_providers


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


def format2runtimes(model_format: Format) -> Tuple:
    if model_format == Format.ONNX:
        return parse_enum(get_available_onnx_providers(), RuntimeProvider)
    elif model_format in (Format.TORCHSCRIPT, Format.TORCH_TRT):
        return (RuntimeProvider.PYT,)
    elif model_format in (Format.TF_SAVEDMODEL, Format.TF_TRT):
        return (RuntimeProvider.TF,)
    elif model_format == Format.TENSORRT:
        return (RuntimeProvider.TRT,)
    else:
        return ()


class Status(str, Parameter):
    OK = "OK"
    FAIL = "FAIL"
    NOOP = "NOOP"
    INITIALIZED = "INITIALIZED"
    SKIPPED = "SKIPPED"


class Framework(Parameter):
    TF2 = "tensorflow2"
    PYT = "pytorch"
    ONNX = "onnx"


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


def get_default_model_name():
    return "navigator_model"


def get_default_status_filename():
    return "status.yaml"


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
    return workdir / f"{model_name}.nav.workspace"


# pytype: disable=wrong-arg-types
def format_to_relative_model_path(
    format: Format, jit_type: JitType = JitType.SCRIPT, precision: TensorRTPrecision = TensorRTPrecision.FP32
) -> Path:
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


def sample_to_tuple(input: Any) -> Tuple[Any, ...]:
    if isinstance(input, Sequence):
        return tuple(input)
    if isinstance(input, Mapping):
        return tuple(input.values())
    return (input,)


def extract_bs1(sample: Sample, batch_dim: Optional[int]) -> Sample:
    if batch_dim is not None:
        return {name: tensor.take([0], batch_dim) for name, tensor in sample.items()}
    return sample


def parse_enum(value: Any, enum_type: Type[T]) -> Tuple[T, ...]:
    if value is not None:
        value = tuple(value) if isinstance(value, (tuple, list)) else (value,)
        value = tuple(enum_type(v) for v in value)
        return value
    return ()


def get_framework_export_formats(framework: Framework):
    return {
        Framework.PYT: {Format.TORCHSCRIPT, Format.ONNX},
        Framework.TF2: {
            Format.TF_SAVEDMODEL,
        },
        Framework.ONNX: {Format.ONNX},
    }[framework]


def get_base_format(format: Format, framework: Framework):
    return {
        Framework.PYT: {
            Format.TENSORRT: Format.ONNX,
            Format.TORCH_TRT: Format.TORCHSCRIPT,
        },
        Framework.TF2: {
            Format.ONNX: Format.TF_SAVEDMODEL,
            Format.TENSORRT: Format.TF_SAVEDMODEL,
            Format.TF_TRT: Format.TF_SAVEDMODEL,
        },
        Framework.ONNX: {Format.TENSORRT: Format.ONNX},
    }[framework].get(format)


def is_tensor(tensor, framework: Framework):
    if framework == Framework.PYT:
        import torch  # pytype: disable=import-error

        return torch.is_tensor(tensor) or isinstance(tensor, numpy.ndarray)
    elif framework == Framework.TF2:
        import tensorflow  # pytype: disable=import-error

        return tensorflow.is_tensor(tensor) or isinstance(tensor, numpy.ndarray)
    else:
        return isinstance(tensor, numpy.ndarray)


def get_tensor_type_name(framework: Framework):
    if framework == Framework.PYT:
        return "Union[torch.Tensor, numpy.ndarray]"
    elif framework == Framework.TF2:
        return "Union[tensorflow.Tensor, numpy.ndarray]"
    else:
        return "numpy.ndarray"


def validate_sample_input(sample, framework: Framework):
    def is_valid(sample):
        if isinstance(sample, Sequence):
            for tensor in sample:
                if not is_tensor(tensor, framework):
                    return False
        elif isinstance(sample, Mapping):
            for tensor in sample.values():
                if not is_tensor(tensor, framework):
                    return False
        else:
            tensor = sample
            if not is_tensor(tensor, framework):
                return False
        return True

    if not is_valid(sample):
        tensor_type = get_tensor_type_name(framework)
        raise UserError(
            f"Invalid sample type. Sample must be of type Union[{tensor_type}, Sequence[{tensor_type}], Mapping[str, {tensor_type}]]. Dataloader returned {sample}."
        )


def validate_sample_output(sample, framework: Framework):
    def is_valid(sample):
        if isinstance(sample, Sequence):
            for tensor in sample:
                if not is_tensor(tensor, framework):
                    return False
        elif isinstance(sample, Mapping):
            for tensor in sample.values():
                if not is_tensor(tensor, framework):
                    return False
        else:
            tensor = sample
            if not is_tensor(tensor, framework):
                return False
        return True

    if not is_valid(sample):
        tensor_type = get_tensor_type_name(framework)
        raise UserError(
            f"Invalid model output type. Output must be of type Union[{tensor_type}, Sequence[{tensor_type}]], Mapping[str, {tensor_type}]]. Model returned {sample}."
        )
