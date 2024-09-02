# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Convert ExportedProgram model to Torch-TensorRT model."""

import pathlib
from typing import Any, Dict, Optional

import fire
import numpy as np
import torch  # pytype: disable=import-error
from loguru import logger

from model_navigator.configuration import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.configuration.device import map_device_string
from model_navigator.core.dataloader import load_samples
from model_navigator.core.tensor import TensorMetadata
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.frameworks.tensorrt.timing_tactics import TimingCacheManager, trt_cache_inplace_cache_dir
from model_navigator.utils.common import numpy_to_torch_dtype


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
    input_metadata: Dict[str, Any],
    shapes: Dict[str, Dict[str, int]],
    batch_dim: Optional[int],
    max_workspace_size: int,
    precision: str,
    precision_mode: str,
    target_device: str,
    debug: bool,
    custom_args: Dict[str, Any],
    timing_cache_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    navigator_workspace: Optional[str] = None,
) -> None:
    """Run conversion from ExportedProgram to Torch-TensorRT.

    Args:
        exported_model_path: ExportedProgram model path.
        converted_model_path: Output Torch-TensorRT model path.
        input_metadata: List of input metadata.
        shapes: Dictionary with min, opt, max shapes of the inputs.
            The key is an input name and the value is a dictionary with keys ("min", "opt", "max")
            and respective values.
        batch_dim: Batch dimension.
        max_workspace_size: Maximum workspace size in bytes.
        precision: TensorRT precision. Could be "fp16" or "fp32".
        precision_mode: TensorRT precision mode.
        target_device: Device on which perform the conversion
        debug: If True print debug logs.
        custom_args: Dictionary with passthrough parameters. For available arguments check PyTorch
                     documentation: https://pytorch.org/TensorRT/py_api/torch_tensorrt.html
        timing_cache_dir: Directory to save timing cache. Defaults to None which means it will be saved in workspace root.
        model_name: Model name for the timing cache. Defaults to None which means it will be named after the model file.
        navigator_workspace: Model Navigator workspace path. When None use current workdir. Defaults to None.
    """
    import torch_tensorrt  # pytype: disable=import-error

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    input_metadata = TensorMetadata.from_json(input_metadata)
    input_dtypes = [tensorrt_utils.cast_type(input_spec.dtype).name for input_spec in input_metadata.values()]

    logger.info(f"Shapes types: {type(shapes)}, Shapes: {shapes}")
    logger.info(f"Input dtypes types: {type(input_dtypes)}, Input dtypes: {input_dtypes}")

    conversion_sample = load_samples("conversion_samples", navigator_workspace, batch_dim)[0]

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
        exported_model_path = navigator_workspace / exported_model_path

    if model_name is None:
        model_name = navigator_workspace.stem

    model = torch.export.load(exported_model_path.as_posix())

    if debug:
        log_level = torch_tensorrt.logging.Level.Debug
        logger.info(f"Logging set to `debug` ({log_level})")
        torch_tensorrt.logging.set_reportable_log_level(log_level)

    target_device = map_device_string(target_device)

    # saving timing cache in model_navigator workspace or ...
    timing_cache = trt_cache_inplace_cache_dir()
    if timing_cache_dir is not None:
        timing_cache = pathlib.Path(timing_cache_dir)

    with TimingCacheManager(model_name=model_name, cache_path=timing_cache_dir) as timing_cache:
        timing_cache_path = timing_cache.as_posix() if timing_cache else None

        # reusing custom_args as dynamo.compile has a default cache path argument
        if timing_cache_path is not None:
            custom_args["timing_cache_path"] = timing_cache_path

        tr_model_compiled = torch_tensorrt.dynamo.compile(
            exported_program=model,
            inputs=model_input_shapes,
            workspace_size=max_workspace_size,
            device=target_device,
            **_get_precision(precision, precision_mode),
            **custom_args,
        )

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = navigator_workspace / converted_model_path

    inputs = []
    for _, val in conversion_sample.items():
        inputs.append(torch.from_numpy(val).to(target_device))

    torch_tensorrt.save(tr_model_compiled, converted_model_path.as_posix(), inputs=inputs)


if __name__ == "__main__":
    fire.Fire(convert)
