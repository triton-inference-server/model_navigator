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

import onnx_graphsurgeon as gs  # pytype: disable=import-error
import onnx

from model_navigator.log import set_logger

LOGGER = logging.getLogger(__name__)


def _remove_idx(node):
    node.name, *idx = node.name.split(":")
    assert len(idx) == 0 or (len(idx) == 1 and idx[0] == "0")
    return node


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Remove :0 idx added by tf2onnx by default for each input")
    parser.add_argument("input_onnx_model", help="Path to input ONNX model")
    parser.add_argument("output_onnx_model", help="Path to input ONNX model")
    parser.add_argument("-v", "--verbose", default=0, action="count", help="Verbose output")
    args = parser.parse_args()

    set_logger(verbose=bool(args.verbose))

    LOGGER.debug(f"Loading {args.input_onnx_model}")
    graph = gs.import_onnx(onnx.load(args.input_onnx_model))
    LOGGER.info(f"Initial inputs: {', '.join([node.name for node in graph.inputs])}")
    graph.inputs = [_remove_idx(node) for node in graph.inputs]
    LOGGER.info(f"Without idx inputs: {', '.join([node.name for node in graph.inputs])}")
    LOGGER.debug(f"Saving {args.input_onnx_model}")
    onnx.save(gs.export_onnx(graph), args.output_onnx_model)


if __name__ == "__main__":
    main()
