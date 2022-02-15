#!/usr/bin/env python3
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
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

# pytype: disable=import-error
import torch

# pytype: enable=import-error
from model_navigator.cli.spec import parse_shapes, parse_value_ranges
from model_navigator.log import log_dict, set_logger, set_tf_verbosity
from model_navigator.model import ModelSignatureConfig
from model_navigator.utils.signature import load_annotation

from .utils import numpy_to_torch_type

LOGGER = logging.getLogger("ts2onnx")


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
    model.eval()

    # TODO: on which device (which cuda idx) we should run conversion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    LOGGER.debug(f"Model is on {device} device")

    value_ranges = parse_value_ranges(None, None, args.value_ranges)
    shapes = parse_shapes(None, None, args.shapes)
    LOGGER.debug(f"Parsed shapes: {shapes}")
    LOGGER.debug(f"Parsed value ranges: {value_ranges}")

    io_spec: ModelSignatureConfig = load_annotation(input_model_path)

    def _extract_dyn_axes(spec):
        indexes = np.where(np.array(spec.shape) == -1)[0]
        return list(map(int, indexes))

    inputs = io_spec.inputs or {}
    outputs = io_spec.outputs or {}
    dynamic_axes: Dict[str, Iterable[int]] = {
        **{spec.name: _extract_dyn_axes(spec) for name, spec in inputs.items()},
        **{spec.name: _extract_dyn_axes(spec) for name, spec in outputs.items()},
    }

    def _extend_dyn_axes(name, shape):
        user_provided_shape = shapes.get(name)
        if user_provided_shape:
            return user_provided_shape
        else:
            max_dim = max(16, max(shape))
            return tuple(dim if dim != -1 else np.random.randint(1, max_dim + 1) for dim in shape)

    def _generate_random(spec_, dtype, device):
        value_range = value_ranges.get(spec_.name, (0, 1))

        size = (*_extend_dyn_axes(spec_.name, spec_.shape),)
        if dtype.kind == "i":
            return torch.randint(value_range[0], value_range[1], size=size, device=device).type(
                numpy_to_torch_type(spec_.dtype.type)
            )
        elif dtype.kind == "f":
            return (
                torch.randn(size=size, device=device).type(numpy_to_torch_type(spec_.dtype.type))
                * (value_range[1] - value_range[0])
                + value_range[0]
            )
        else:
            raise ValueError(f"Don't know how to generate random tensor for dtype={dtype}")

    # FIXME: argument handling here is really awful, unintuitive and can break easily.
    # Ideally, Triton could support kwargs in the model's forward method,
    # which has been recently implemented in libtorch.
    input_names = [name for name, spec in inputs.items()]
    output_names = [name for name, spec in outputs.items()]

    LOGGER.info(f"Dynamic axes indexes: {dynamic_axes}")

    dummy_input = [_generate_random(spec, spec.dtype, device) for spec in inputs.values()]

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=args.onnx_path,
        opset_version=args.opset_version,
        verbose=args.verbose,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    main()
