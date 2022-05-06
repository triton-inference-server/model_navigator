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
from functools import singledispatch
from pathlib import Path
from typing import Optional

# pytype: disable=import-error
import torch
import torch_tensorrt as trtorch

from model_navigator.converter.config import TensorRTConversionConfig, TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.converter.dataloader import Dataloader
from model_navigator.converter.pyt.utils import numpy_to_torch_type
from model_navigator.exceptions import ModelNavigatorConverterException
from model_navigator.model import ModelSignatureConfig

# pytype: enable=import-error


LOGGER = logging.getLogger("torch_tensorrt_adapter")


def _get_precision(precision, precision_mode):
    if precision_mode == TensorRTPrecisionMode.HIERARCHY:
        enabled_precisions = {
            TensorRTPrecision.FP32: [torch.float],
            TensorRTPrecision.TF32: [torch.float],
            TensorRTPrecision.FP16: [torch.float, torch.half],
            TensorRTPrecision.INT8: [torch.float, torch.half, torch.int8],
        }[precision]
    elif precision_mode == TensorRTPrecisionMode.SINGLE:
        enabled_precisions = {
            TensorRTPrecision.FP32: [torch.float],
            TensorRTPrecision.TF32: [torch.float],
            TensorRTPrecision.FP16: [torch.half],
            TensorRTPrecision.INT8: [torch.int8],
        }[precision]
    else:
        raise ModelNavigatorConverterException(
            f"Unsupported precision mode {precision_mode}. Only {TensorRTPrecisionMode.HIERARCHY} and "
            "{TensorRTPrecisionMode.SINGLE} are allowed"
        )
    return {
        "enabled_precisions": enabled_precisions,
        "disable_tf32": precision == TensorRTPrecision.FP32,
    }


def _cast_down(dtype):
    if dtype == torch.int64:
        return torch.int32
    elif dtype == torch.float64:
        return torch.float32
    return dtype


def _trtorch_inputs(dataloader):
    ret = []

    for name, dtype in dataloader.dtypes.items():
        dtype = _cast_down(numpy_to_torch_type(dtype))
        shapes = {
            "min_shape": dataloader.min_shapes[name],
            "opt_shape": dataloader.opt_shapes[name],
            "max_shape": dataloader.max_shapes[name],
        }

        # Workaround for torch-trt issues with resnet50 (as of 22.02).
        # If the shapes differ only on the batch dimension,
        # then use a single static shape - this significantly increases
        # the chances of success when compiling to TensorRT.
        # Hopefully, this can be removed in the future
        if all(x[1:] == y[1:] for x in shapes.values() for y in shapes.values()):
            max_batch_size = max(x[0] for x in shapes.values())
            sample_shape = next(iter(shapes.values()))[1:]
            shapes = {"shape": (max_batch_size, *sample_shape)}

        ret.append(
            trtorch.Input(dtype=dtype, **shapes),
        )

    return ret


@singledispatch
def load_model(model) -> torch.nn.Module:
    logging.error(f"Got {model} as input to _load_model")
    raise TypeError()


@load_model.register(Path)
def _load_model_from_path(input_model: Path):
    device = torch.device("cuda")
    model = torch.jit.load(input_model.as_posix())
    model.to(device).eval()
    LOGGER.debug(f"Model is on {device} device")

    return model


@load_model.register(torch.nn.Module)
def _load_model_noop(input_model: torch.nn.Module):
    return input_model


@singledispatch
def save_model(model, output_path):
    raise TypeError()


@save_model.register(bytes)
def _save_trt_plan(model: bytes, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fp:
        fp.write(model)


@save_model.register(torch.jit.ScriptModule)
def _save_module(model: torch.jit.ScriptModule, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path.as_posix())


def convert_to_trt_engine(
    *,
    input_model,  # : Union[torch.nn.Module, Path],
    output_path: Path,
    log_path: Path,
    signature_config: Optional[ModelSignatureConfig],
    tensorrt_config: TensorRTConversionConfig,
    dataloader: Dataloader,
):
    """Convert TorchScript model from file at `input_path` to a TensorRT model
    and store it at `output_path`."""
    model = load_model(input_model)
    out = trtorch.ts.convert_method_to_trt_engine(
        model,
        "forward",
        inputs=_trtorch_inputs(dataloader),
        strict_types=tensorrt_config.strict_types,
        sparse_weights=tensorrt_config.sparse_weights,
        workspace_size=tensorrt_config.max_workspace_size,
        truncate_long_and_double=True,
        **_get_precision(tensorrt_config.precision, tensorrt_config.precision_mode),
    )
    save_model(out, output_path)


def compile(
    *,
    input_model,  #: Union[Tuple[torch.nn.Module, ModelSignatureConfig], Path]
    output_path: Path,
    log_path: Path,
    signature_config: Optional[ModelSignatureConfig],
    tensorrt_config: TensorRTConversionConfig,
    dataloader: Dataloader,
):
    """Convert TorchScript model from file at `input_path` to a TensorRT model
    and store it at `output_path`."""
    model = load_model(input_model)
    out = trtorch.ts.compile(
        model,
        inputs=_trtorch_inputs(dataloader),
        strict_types=tensorrt_config.strict_types,
        sparse_weights=tensorrt_config.sparse_weights,
        workspace_size=tensorrt_config.max_workspace_size,
        truncate_long_and_double=True,
        **_get_precision(tensorrt_config.precision, tensorrt_config.precision_mode),
    )
    save_model(out, output_path)
