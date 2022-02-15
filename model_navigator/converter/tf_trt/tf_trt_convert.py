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
from typing import Dict, Optional, Tuple

# pytype: disable=import-error
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trtc

from model_navigator.cli.spec import parse_shapes, parse_value_ranges
from model_navigator.converter.tf import utils as tf_utils
from model_navigator.log import log_dict, set_tf_verbosity

# pytype: enable=import-error
LOGGER = logging.getLogger("tf_trt_converter")


def convert_tf2(
    input_path: Path,
    output_path: Path,
    max_workspace_size: int,
    minimum_segment_size: int,
    precision: str,
    max_batch_size: Optional[int],
    shapes: Optional[Dict[str, Dict[str, Tuple]]],
    value_ranges: Optional[Dict[str, Tuple]],
):
    """Optimize a Tensorflow 2.x Savedmodel at `input_path`
    with TF-TRT.
    Store the resulting SavedModel at `output_path`.
    """
    if precision.lower() == "tf32":
        LOGGER.info("Precision TF32 is equivalent to FP32")
        precision = "fp32"

    params = trtc.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=max_workspace_size,
        precision_mode=precision,
        minimum_segment_size=minimum_segment_size,
    )
    # TODO: check if model signature has dynamic shapes
    # TODO: allow setting dynamic_shape_profile_strategy
    converter = trtc.TrtGraphConverterV2(
        input_saved_model_dir=input_path.as_posix(),
        conversion_params=params,
        use_dynamic_shape=True,
    )

    concrete_func = converter.convert()
    if not shapes and max_batch_size:
        shapes = tf_utils.get_default_profile(concrete_func, max_batch_size)

    if shapes:
        LOGGER.info("Pre-building TRT engines.")
        inputs = tf_utils.generate_inputs(concrete_func, shapes, value_ranges)
        converter.build(inputs)

    converter.save(output_path.as_posix())


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="TorchScript to TRT coverter")
    parser.add_argument("--input-path", help="Path to TF-Savedmodel input model", type=Path)
    parser.add_argument("--output-path", help="Path to TF-TRT output model", type=Path)
    parser.add_argument(
        "--max-workspace-size",
        type=int,
        help="Maximum size of workspace given to TensorRT.",
    )
    parser.add_argument(
        "--min-segment-size",
        type=int,
        help="Minimum number of nodes required for a subgraph to be replaced by TRTEngineOp.",
        default=3,
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="Maximum batch size to build TRT engines for.",
    )
    parser.add_argument(
        "--trt-min-shapes",
        nargs="*",
        type=str,
        help="Format: --trt-min-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN",
    )
    parser.add_argument(
        "--trt-max-shapes",
        nargs="*",
        type=str,
        help="Format: --trt-max-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN",
    )
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
        "--precision",
        type=str,
        choices=["TF32", "FP32", "FP16", "INT8", "tf32", "fp32", "fp16", "int8"],
        default="FP32",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def _setup_logging(args):
    set_tf_verbosity(verbose=args.verbose)
    log_dict("args", vars(args))


def _main():
    # don't allow TF to preallocate gpu memory, because TF-TRT
    # will not have enough (it seem to allocate some outside the TF pool).
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    args = _parse_args()
    _setup_logging(args)

    shapes = None
    if args.trt_min_shapes and args.trt_max_shapes:
        shapes = {
            "min": parse_shapes(None, None, args.trt_min_shapes),
            "max": parse_shapes(None, None, args.trt_max_shapes),
        }
        log_dict("Prepared shapes:", shapes)
    elif not args.max_batch_size:
        LOGGER.info(
            "Neither max_batch_size, nor a full dataset profile provided. "
            "TRT engines will not be built at conversion time."
        )

    convert_tf2(
        input_path=args.input_path,
        output_path=args.output_path,
        max_workspace_size=args.max_workspace_size,
        minimum_segment_size=args.min_segment_size,
        precision=args.precision,
        shapes=shapes,
        max_batch_size=args.max_batch_size,
        value_ranges=parse_value_ranges(None, None, args.value_ranges),
    )


if __name__ == "__main__":
    _main()
