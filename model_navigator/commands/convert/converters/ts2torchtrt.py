# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Convert TorchScript model to Torch-TensorRT model."""

import logging
import pathlib
from typing import Any, Dict, List, Optional

import fire
import numpy as np
import torch  # pytype: disable=import-error

from model_navigator.api.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.utils.common import numpy_to_torch_dtype

LOGGER = logging.getLogger(__name__)


def _get_precision(precision, precision_mode):
    precision = TensorRTPrecision(precision)
    precision_mode = TensorRTPrecisionMode(precision_mode)
    if precision_mode == TensorRTPrecisionMode.HIERARCHY:
        enabled_precisions = {
            TensorRTPrecision.FP32: [torch.float],
            TensorRTPrecision.FP16: [torch.float, torch.half],
            TensorRTPrecision.INT8: [torch.float, torch.half, torch.int8],
        }[precision]
    elif precision_mode == TensorRTPrecisionMode.SINGLE:
        enabled_precisions = {
            TensorRTPrecision.FP32: [torch.float],
            TensorRTPrecision.FP16: [torch.half],
            TensorRTPrecision.INT8: [torch.int8],
        }[precision]
    else:
        raise ValueError(
            f"Unsupported precision mode {precision_mode}. Only {TensorRTPrecisionMode.HIERARCHY} and "
            f"{TensorRTPrecisionMode.SINGLE} are allowed"
        )
    return {
        "enabled_precisions": enabled_precisions,
    }


def convert(
    exported_model_path: str,
    converted_model_path: str,
    shapes: Dict[str, Dict[str, int]],
    input_dtypes: List[str],
    max_workspace_size: int,
    precision: str,
    precision_mode: str,
    target_device: str,
    debug: bool,
    custom_args: Dict[str, Any],
    workspace: Optional[str] = None,
) -> None:
    """Run conversion from TorchScript to Torch-TensorRT.

    Args:
        exported_model_path (str): TorchScript model path.
        converted_model_path (str): Output Torch-TensorRT model path.
        shapes (Dict[str, Dict[str, int]]): Dictionary with min, opt, max shapes of the inputs.
            The key is an input name and the value is a dictionary with keys ("min", "opt", "max")
            and respective values.
        input_dtypes (List[str]): List of inputs data types.
        max_workspace_size (int): Maximum workspace size in bytes.
        precision (str): TensorRT precision. Could be "fp16" or "fp32".
        precision_mode (str): TensorRT precision mode.
        target_device (str): _description_
        debug (bool): If True print debug logs.
        workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        custom_args (Optional[Dict[str, str]], optional): Dictionary with passthrough parameters.
            For available arguments check PyTorch documentation: https://pytorch.org/TensorRT/py_api/torch_tensorrt.html
    """
    import torch_tensorrt  # pytype: disable=import-error

    if not workspace:
        workspace = pathlib.Path.cwd()
    workspace = pathlib.Path(workspace)

    LOGGER.info(f"Shapes types: {type(shapes)}, Shapes: {shapes}")
    LOGGER.info(f"Input dtypes types: {type(input_dtypes)}, Input dtypes: {input_dtypes}")

    input_dtypes = [numpy_to_torch_dtype(np.dtype(input_dtype)) for input_dtype in input_dtypes]
    model_input_shapes = []
    for input_shapes, input_dtype in zip(shapes.values(), input_dtypes):
        model_input_shapes.append(
            torch_tensorrt.Input(
                min_shape=input_shapes["min"],
                opt_shape=input_shapes["opt"],
                max_shape=input_shapes["max"],
                dtype=input_dtype,
            )
        )

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = workspace / exported_model_path

    model = torch.jit.load(exported_model_path.as_posix(), map_location=target_device)

    if debug:
        log_level = torch_tensorrt.logging.Level.Debug
        LOGGER.info(f"Logging set to `debug` ({log_level})")
        torch_tensorrt.logging.set_reportable_log_level(log_level)

    tr_model_compiled = torch_tensorrt.compile(
        module=model,
        inputs=model_input_shapes,
        workspace_size=max_workspace_size,
        truncate_long_and_double=True,
        **_get_precision(precision, precision_mode),
        **custom_args,
    )

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = workspace / converted_model_path

    tr_model_compiled.save(converted_model_path.as_posix())


if __name__ == "__main__":
    fire.Fire(convert)
