#!/usr/bin/env python3
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
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

# pytype: disable=import-error
import torch

# pytype: enable=import-error
from model_navigator.log import set_logger, set_tf_verbosity, log_dict
from model_navigator.tensor import IOSpec, TensorSpec

LOGGER = logging.getLogger("ts2onnx")

NUMPY_TO_TORCH_DTYPE_DICT = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def _parse_value_range(value_ranges):
    results = {}
    for value_range in value_ranges:
        name, range_str = value_range.split(":")
        min_value, max_value = range_str.split(",")

        has_dot = "." in min_value or "." in max_value
        min_value, max_value = float(min_value), float(max_value)
        if min_value.is_integer() and max_value.is_integer() and not has_dot:
            min_value, max_value = int(min_value), int(max_value)

        results[name] = (min_value, max_value)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TorchScript to ONNX coverter")
    parser.add_argument("torchscript_path", help="Path to TorchScript input model")
    parser.add_argument("onnx_path", help="Path to ONNX output model")
    parser.add_argument("--opset-version", type=int, help="Version of ONNX opset to use")
    parser.add_argument(
        "--value-ranges",
        nargs="*",
        type=str,
        help=(
            "Range of values used during performance analysis defined per input. "
            "Format: --value-range input_name0:min_value,max_value .. input_nameN:min_value,max_value"
        ),
    )
    parser.add_argument(
        "--shapes",
        nargs="*",
        type=str,
        help="Shapes for dynamic axes. Format: --shapes <input0>:D0,D1,..,DN .. <inputN>:D0,D1,..,DN",
    )
    parser.add_argument("-v", "--verbose", default=0, action="count", help="Verbose output")
    args = parser.parse_args()

    set_logger(verbose=args.verbose)
    set_tf_verbosity(verbose=args.verbose)
    log_dict("args", vars(args))

    input_model_path = Path(args.torchscript_path)
    model = torch.jit.load(input_model_path.as_posix())
    value_ranges = _parse_value_range(args.value_ranges)
    shapes = {spec.name: spec for spec in [TensorSpec.from_command_line(s) for s in args.shapes]}

    annotation_path = input_model_path.parent / f"{input_model_path.stem}.yaml"
    io_spec = IOSpec.from_file(annotation_path)
    print("shapes", shapes)

    def _extract_dyn_axes(spec):
        indexes = np.where(np.array(spec.shape) == -1)[0]
        return list(map(int, indexes))

    dynamic_axes: Dict[str, Iterable[int]] = {
        **{spec.name: _extract_dyn_axes(spec) for name, spec in io_spec.inputs.items()},
        **{spec.name: _extract_dyn_axes(spec) for name, spec in io_spec.outputs.items()},
    }

    def _extend_dyn_axes(name, shape):
        user_provided_shape = shapes.get(name)
        if user_provided_shape:
            return user_provided_shape.shape
        else:
            max_dim = max(16, max(shape))
            return tuple([dim if dim != -1 else np.random.randint(1, max_dim + 1) for dim in shape])

    def _generate_random(spec_, dtype):
        min_value, max_value = value_ranges.get(spec_.name, (0, 1))

        size = (*_extend_dyn_axes(spec_.name, spec_.shape),)
        device = "cuda"
        if dtype.kind == "i":
            return torch.randint(min_value, max_value, size=size, device=device).type(
                NUMPY_TO_TORCH_DTYPE_DICT[spec_.dtype.type]
            )
        elif dtype.kind == "f":
            return (
                torch.randn(size=size, device=device).type(NUMPY_TO_TORCH_DTYPE_DICT[spec_.dtype.type])
                * (max_value - min_value)
                + min_value
            )
        else:
            raise ValueError(f"Don't know how to generate random tensor for dtype={dtype}")

    dummy_input = {spec.name: _generate_random(spec, spec.dtype) for name, spec in io_spec.inputs.items()}
    input_names = [name for name, spec in io_spec.inputs.items()]
    output_names = [name for name, spec in io_spec.outputs.items()]

    LOGGER.info("Model args:")
    for name, tensor in dummy_input.items():
        LOGGER.info(f"    - name={name} shape={tensor.shape} dtype={tensor.dtype} device={tensor.device}")
        LOGGER.info(f"      {tensor}")
    LOGGER.info(f"Dynamic axes indexes: {dynamic_axes}")

    torch.onnx.export(
        model=model,
        args=(dummy_input,),
        f=args.onnx_path,
        opset_version=args.opset_version,
        verbose=args.verbose,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        enable_onnx_checker=True,
    )


if __name__ == "__main__":
    main()
