# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
from typing import Any, Dict, List, Optional

import fire
import numpy as np
import torch  # pytype: disable=import-error
from loguru import logger
from packaging.version import Version

from model_navigator.configuration import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.configuration.device import map_device_string
from model_navigator.core.dataloader import expand_sample, load_samples
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.frameworks.tensorrt.timing_tactics import TimingCacheManager
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
    shapes: Dict[str, Dict[str, List[int]]],
    batch_dim: Optional[int],
    max_workspace_size: int,
    pickle_protocol: int,
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
        pickle_protocol: Pickle protocol used during model serialization
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

    if batch_dim is None:
        max_batch_size = None
        expanded_sample = expand_sample(conversion_sample, input_metadata, batch_dim=batch_dim, batch_size=None)
    else:
        # WAR to make data dynamic
        max_batch_size = list(shapes.values())[0]["max"][0]
        batch_size = 2 if max_batch_size > 1 else 1  # select the minimum value to expand samples
        expanded_sample = expand_sample(conversion_sample, input_metadata, batch_dim=batch_dim, batch_size=batch_size)

    dummy_input = {n: torch.from_numpy(val).to(target_device) for n, val in expanded_sample.items()}
    dummy_input = input_metadata.unflatten_sample(dummy_input, wrap_input=False)

    if not isinstance(dummy_input, tuple):
        dummy_input = (dummy_input,)
    if not isinstance(dummy_input[-1], dict):
        dummy_input = (*dummy_input, {})
    *args, kwargs = dummy_input

    input_dtypes = [numpy_to_torch_dtype(np.dtype(input_dtype)) for input_dtype in input_dtypes]
    model_input_shapes = []
    dynamic_shapes = []
    for input_name, input_dtype in zip(shapes.keys(), input_dtypes):
        input_shapes = shapes.get(input_name)
        tensor_metadata = input_metadata.get(input_name)
        if not tensor_metadata or not input_shapes:
            LOGGER.warning(f"Input metadata or input shapes for input {input_name} is not found")
            continue

        model_input_shapes.append(
            torch_tensorrt.Input(
                min_shape=input_shapes["min"],
                opt_shape=input_shapes["opt"],
                max_shape=input_shapes["max"],
                dtype=input_dtype,
            )
        )

        dynamic_shape_map = {}
        if max_batch_size is not None and max_batch_size > 1 and len(tensor_metadata.shape) > 0:
            dynamic_shape_map[0] = torch.export.Dim(f"{input_name}_batch", min=1, max=max_batch_size)

        for idx in range(1, len(input_shapes["min"])):
            min_value = input_shapes["min"][idx]
            max_value = input_shapes["max"][idx]
            if min_value != max_value:
                dynamic_shape_map[idx] = torch.export.Dim(f"{input_name}__{idx}", min=min_value, max=max_value)

        dynamic_shapes.append(dynamic_shape_map)

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

    with TimingCacheManager(model_name=model_name, cache_path=timing_cache_dir) as timing_cache:
        timing_cache_path = timing_cache.as_posix() if timing_cache else None

        # reusing custom_args as dynamo.compile has a default cache path argument
        if timing_cache_path is not None:
            custom_args["timing_cache_path"] = timing_cache_path

        trt_model_compiled = torch_tensorrt.dynamo.compile(
            exported_program=model,
            inputs=model_input_shapes,
            workspace_size=max_workspace_size,
            device=target_device,
            **_get_precision(precision, precision_mode),
            **custom_args,
        )

        exported_model = torch.export.export(
            trt_model_compiled,
            args=tuple(args),
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = navigator_workspace / converted_model_path

    save_kwargs = {}
    if Version(torch.__version__) > Version("2.6"):
        LOGGER.info("Using pickle protocol {}.", pickle_protocol)
        save_kwargs["pickle_protocol"] = pickle_protocol

    torch.export.save(exported_model, converted_model_path.as_posix(), **save_kwargs)


if __name__ == "__main__":
    fire.Fire(convert)
