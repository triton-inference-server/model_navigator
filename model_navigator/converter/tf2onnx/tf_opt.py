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
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# pytype: disable=import-error
import tensorflow as tf
from tf2onnx import tf_loader, utils

from model_navigator.converter.tf2onnx.tf_saver import to_savedmodel
from model_navigator.converter.tf.utils import obtain_inputs
from model_navigator.tensor import TensorSpec

# pytype: enable=import-error


LOGGER = logging.getLogger(__name__)


def handle_tensor_specs(
    graph_def, inputs: Dict[str, str], outputs: Dict[str, str]
) -> Tuple[Dict[str, TensorSpec], Dict[str, TensorSpec]]:
    def tensor2tensor_spec(tensor):
        shape = tuple(s.value if hasattr(s, "value") else s for s in tensor.shape)
        shape = tuple(dim if dim is not None else -1 for dim in shape)
        return TensorSpec(name=tensor.name, dtype=np.dtype(tensor.dtype.name), shape=shape)

    with tf.device("/cpu:0"):
        session_config = tf.compat.v1.ConfigProto(graph_options=tf.compat.v1.GraphOptions(infer_shapes=True))
        tf_loader.tf_reset_default_graph()
        with tf_loader.tf_session(config=session_config) as sess:
            tf.import_graph_def(graph_def, name="")

            def _get_spec(tensors_dict):
                tensors_dict = {name: sess.graph.get_tensor_by_name(tname) for name, tname in tensors_dict.items()}
                return {name: tensor2tensor_spec(tensor) for name, tensor in tensors_dict.items()}

            inputs = _get_spec(inputs)
            outputs = _get_spec(outputs)

        tf_loader.tf_reset_default_graph()
    return inputs, outputs


def _obtain_inputs(input_names, concrete_func_and_imported):
    if tf_loader.is_tf2():
        concrete_func, imported = concrete_func_and_imported
        inputs = obtain_inputs(concrete_func)
    else:
        raise NotImplementedError()
    return inputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--saved-model", help="input from saved model")
    parser.add_argument("--tag", help="tag to use for saved_model")
    parser.add_argument("--output", help="output model file")
    parser.add_argument("--inputs", help="model input_names (optional for saved_model)")
    parser.add_argument("--outputs", help="model output_names (optional for saved_model)")
    parser.add_argument("--signature_def", help="signature_def from saved_model to use")
    parser.add_argument(
        "--concrete_function",
        type=int,
        default=None,
        help="For TF2.x saved_model, index of func signature in __call__ (--signature_def is ignored)",
    )
    parser.add_argument("--large_model", help="use the large model format (for models > 2GB)", action="store_true")
    parser.add_argument("--verbose", "-v", help="verbouse output", action="store_true")
    args = parser.parse_args()

    if args.inputs:
        args.inputs, args.shape_override = utils.split_nodename_and_shape(args.inputs)
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.signature_def:
        args.signature_def = [args.signature_def]

    if args.saved_model:
        (
            graph_def,
            inputs,
            outputs,
            *concrete_func_and_imported,
            initialized_tables,
            tensors_to_rename,
        ) = tf_loader.from_saved_model(
            args.saved_model,
            args.inputs,
            args.outputs,
            args.tag,
            args.signature_def,
            args.concrete_function,
            args.large_model,
            return_concrete_func=True,
            return_initialized_tables=True,
            return_tensors_to_rename=True,
        )

    inputs = _obtain_inputs(inputs, concrete_func_and_imported)
    outputs = {tensors_to_rename.get(output_name, output_name): output_name for output_name in outputs}

    inputs, outputs = handle_tensor_specs(graph_def, inputs, outputs)

    LOGGER.debug(f"inputs: {inputs}")
    LOGGER.debug(f"outputs: {outputs}")

    if Path(args.output).exists:
        LOGGER.warning(f"Removing old {args.output}")
        shutil.rmtree(args.output, ignore_errors=True)
    to_savedmodel(graph_def, inputs, outputs, args.output)


if __name__ == "__main__":
    main()
