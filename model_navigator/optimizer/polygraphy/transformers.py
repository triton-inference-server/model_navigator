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
from distutils.version import LooseVersion
from pathlib import Path
from typing import Optional, Dict

import polygraphy
import sh

from model_navigator import Format, Precision
from model_navigator.model_navigator_exceptions import ModelNavigatorOptimizerException
from model_navigator.optimizer.polygraphy.utils import Profiles
from model_navigator.optimizer.utils import execute_sh_command, prepare_log_header
from model_navigator.tensor import TensorSpec

LOGGER = logging.getLogger("polygraphy.transformers")
DEFAULT_MAX_WORKSPACE_SIZE = 4 * 2 ** 30  # 4GB

POLYGRAPHY_VERSION = LooseVersion(polygraphy.__version__)
DEFAULT_TOLERANCES = {"rtol": 1e-5, "atol": 1e-5}


def _format_tensor_spec_new(spec, add_dtype=True):
    result = f"{spec.name}:[{','.join(map(str, spec.shape))}]"
    if add_dtype:
        result += f":{spec.dtype}"
    return result


def _format_tensor_spec(spec, add_dtype=True):
    result = f"{spec.name},{'x'.join(map(str, spec.shape))}"
    if add_dtype:
        result += f":{spec.dtype}"
    return result


def _format_tolerance(name, value):
    sep = ":" if POLYGRAPHY_VERSION > LooseVersion("0.24.2") else ","
    return f"{name}{sep}{value}" if name else str(value)


def onnx2trt(
    *,
    input_path: Path,
    output_path: Path,
    precision: Precision,
    max_workspace_size: Optional[int] = None,
    profiles: Optional[Profiles] = None,
    rtol: Optional[Dict[str, float]] = None,
    atol: Optional[Dict[str, float]] = None,
    verbose: bool = False,
):
    import onnx
    from polygraphy.backend.onnx.loader import OnnxFromPath
    from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata
    from sh import polygraphy  # noqa

    LOGGER.debug(f"Using Polygraphy version: {POLYGRAPHY_VERSION}")

    model: onnx.ModelProto = OnnxFromPath(input_path.as_posix())()

    inputs = get_input_metadata(model.graph)
    inputs = [TensorSpec.from_polygraphy_metadata_tuple(name, meta) for name, meta in inputs.items()]

    outputs = get_output_metadata(model.graph)
    outputs = [TensorSpec.from_polygraphy_metadata_tuple(name, meta) for name, meta in outputs.items()]

    trt_precision_flags = {
        Precision.FP32: "--tf32",
        Precision.TF32: "--tf32",
        Precision.FP16: "--fp16",
    }[precision]

    LOGGER.warning("This conversion should be done on target GPU platform")

    profiling_flags = []
    if profiles:
        if profiles.opt_shapes:
            inputs = profiles.opt_shapes

        shapes = [profiles.min_shapes, profiles.min_shapes, profiles.min_shapes]
        if any(shapes) and not all(shapes):
            raise ModelNavigatorOptimizerException(
                "Not all profile parameter provided. Use --[min|opt|max]-shapes to define complete profile."
            )
        elif all(shapes):
            profiling_flags = [
                "--trt-min-shapes",
                *[_format_tensor_spec(spec, add_dtype=False) for spec in profiles.min_shapes],
                "--trt-opt-shapes",
                *[_format_tensor_spec(spec, add_dtype=False) for spec in profiles.opt_shapes],
                "--trt-max-shapes",
                *[_format_tensor_spec(spec, add_dtype=False) for spec in profiles.max_shapes],
            ]

    # TODO: obtain free memory on gpu
    if max_workspace_size is None:
        max_workspace_size = DEFAULT_MAX_WORKSPACE_SIZE
        LOGGER.warning(
            f"--max-workspace-size config parameter is missing thus using {DEFAULT_MAX_WORKSPACE_SIZE}; "
            f"specify this config parameter in case of OOM Error or poor TRT performance."
        )

    tolerance_flags = []

    def _add_tolerance_params(params_name, tolerance_params):
        if tolerance_params:
            tolerance_params.setdefault("", DEFAULT_TOLERANCES[params_name])
            params = [_format_tolerance(name, value) for name, value in tolerance_params.items()]
            tolerance_flags.extend([f"--{params_name}"] + params)

    _add_tolerance_params("rtol", rtol or {})
    _add_tolerance_params("atol", atol or {})

    args = [
        "--onnxrt",
        "--trt",
        input_path.as_posix(),
        "--model-type",
        "onnx",
        "--inputs",
        *[_format_tensor_spec(spec, add_dtype=False) for spec in inputs],
        "--onnx-outputs",
        *[spec.name for spec in outputs],
        "--shape-inference",
        trt_precision_flags,
        *profiling_flags,
        *tolerance_flags,
        "--workspace",
        max_workspace_size,
        "--save-engine",
        output_path,
    ]
    if verbose:
        args += [
            "-v",
        ]
    log_path = output_path.parent / f"{output_path.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.ONNX, Format.TRT)
            execute_sh_command(polygraphy.run.bake(*args), log_file=log_file, verbose=verbose)
        LOGGER.info("Optimization succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"Optimization failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorOptimizerException(e)
