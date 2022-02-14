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
import sys
from functools import singledispatch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import click

# pytype: disable=import-error
import torch
import torch_tensorrt as trtorch

from model_navigator.cli.spec import parse_shapes
from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.converter.pyt.utils import numpy_to_torch_type
from model_navigator.exceptions import ModelNavigatorConverterException
from model_navigator.log import log_dict, set_logger, set_tf_verbosity
from model_navigator.model import ModelSignatureConfig
from model_navigator.utils.signature import load_annotation

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


def _trtorch_inputs(io_spec: ModelSignatureConfig, shapes, max_batch_size=0):
    ret = []
    for name, input_ in io_spec.inputs.items():
        dtype = _cast_down(numpy_to_torch_type(input_.dtype.type))

        if shapes:
            try:
                ret.append(
                    trtorch.Input(
                        min_shape=shapes["min"][name],
                        opt_shape=shapes["opt"][name],
                        max_shape=shapes["max"][name],
                        dtype=dtype,
                    ),
                )
            except TypeError:
                # we have a single shape
                ret.append(trtorch.Input(shape=shapes["max"], dtype=dtype))
        else:
            assert max_batch_size > 0
            shape = list(input_.shape)
            shape[0] = max_batch_size
            ret.append(trtorch.Input(shape=shape, dtype=dtype))

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
    precision: TensorRTPrecision,
    precision_mode: TensorRTPrecisionMode,
    shapes: Union[Dict[str, Dict[str, Tuple]], Dict[str, Tuple]],
    tensorrt_sparse_weights: bool,
    tensorrt_strict_types: bool = False,
    max_batch_size: int = 0,
    max_workspace_size: int = 0,
):
    """Convert TorchScript model from file at `input_path` to a TensorRT model
    and store it at `output_path`."""
    model = load_model(input_model)
    out = trtorch.ts.convert_method_to_trt_engine(
        model,
        "forward",
        inputs=_trtorch_inputs(signature_config, shapes, max_batch_size),
        strict_types=tensorrt_strict_types,
        sparse_weights=tensorrt_sparse_weights,
        workspace_size=max_workspace_size,
        **_get_precision(precision, precision_mode),
    )
    save_model(out, output_path)


def compile(
    *,
    input_model,  #: Union[Tuple[torch.nn.Module, ModelSignatureConfig], Path]
    output_path: Path,
    log_path: Path,
    signature_config: Optional[ModelSignatureConfig],
    precision: TensorRTPrecision,
    precision_mode: TensorRTPrecisionMode,
    shapes: Union[Dict[str, Dict[str, Tuple]], Dict[str, Tuple]],
    tensorrt_sparse_weights: bool,
    tensorrt_strict_types: bool = False,
    max_batch_size: int = 0,
    max_workspace_size: int = 0,
):
    """Convert TorchScript model from file at `input_path` to a TensorRT model
    and store it at `output_path`."""
    model = load_model(input_model)
    out = trtorch.ts.compile(
        model,
        inputs=_trtorch_inputs(signature_config, shapes, max_batch_size),
        strict_types=tensorrt_strict_types,
        sparse_weights=tensorrt_sparse_weights,
        workspace_size=max_workspace_size,
        truncate_long_and_double=True,
        **_get_precision(precision, precision_mode),
    )
    save_model(out, output_path)


def _setup_logging(verbose=False, **kwargs):
    set_logger(verbose=verbose)
    set_tf_verbosity(verbose=verbose)
    log_dict("args", {"verbose": verbose, **kwargs})


@click.command()
@click.option("--input-model", type=Path)
@click.option("--output-path", type=Path)
@click.option("--log-path", type=Path)
@click.option("--precision", type=TensorRTPrecision)
@click.option("--precision-mode", type=TensorRTPrecisionMode)
@click.option("--trt-min-shapes", type=str, default=[], multiple=True)
@click.option("--trt-max-shapes", type=str, default=[], multiple=True)
@click.option("--trt-opt-shapes", type=str, default=[], multiple=True)
@click.option("--tensorrt-sparse-weights", default=False, type=bool)
@click.option("--tensorrt-strict-types", default=False, type=bool)
@click.option("--max-batch-size", default=0, type=int)
@click.option("--max-workspace-size", default=0, type=int)
@click.option("--verbose", default=False, type=bool, is_flag=True)
def main(
    input_model: Path,
    output_path: Path,
    log_path: Path,
    precision: TensorRTPrecision,
    precision_mode: TensorRTPrecisionMode,
    trt_min_shapes: Optional[str],
    trt_max_shapes: Optional[str],
    trt_opt_shapes: Optional[str],
    tensorrt_sparse_weights: bool,
    tensorrt_strict_types: bool,
    max_batch_size: int,
    max_workspace_size: int,
    verbose: bool,
):
    _setup_logging(verbose)

    if not torch.cuda.is_available():
        LOGGER.error("CUDA device not available")
        sys.exit(-1)

    try:
        shapes = {}
        if trt_max_shapes:
            shapes["max"] = parse_shapes(None, None, list(trt_max_shapes))
        if trt_min_shapes:
            shapes["min"] = parse_shapes(None, None, list(trt_min_shapes))
        if trt_opt_shapes:
            shapes["opt"] = parse_shapes(None, None, list(trt_opt_shapes))
    except Exception as e:
        LOGGER.debug(f"parsing shapes failed: {e}")
        shapes = None

    io_spec = load_annotation(input_model)

    if not shapes and not max_batch_size:
        LOGGER.error("Neither max_batch_size, nor a full dataset profile provided. Aborting.")
        sys.exit(-1)

    compile(
        input_model=input_model,
        output_path=output_path,
        log_path=log_path,
        signature_config=io_spec,
        precision=precision,
        precision_mode=precision_mode,
        shapes=shapes,
        tensorrt_sparse_weights=tensorrt_sparse_weights,
        tensorrt_strict_types=tensorrt_strict_types,
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
    )


if __name__ == "__main__":
    main()
