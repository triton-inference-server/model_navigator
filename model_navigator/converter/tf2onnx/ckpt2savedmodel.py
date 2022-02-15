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

from typing import Any, Dict

# pytype: disable=import-error
import tensorflow as tf
from tensorflow.python.framework import graph_util, meta_graph
from tf2onnx.shape_inference import infer_shape
from tf2onnx.tf_loader import from_checkpoint, is_function, tf_session

# pytype: enable=import-error

# TODO: uncomment for graphdef
# def from_checkpoint2(model_path, input_names, output_names):
#     print("xxxxx signature", meta_graph_def.signature_def)
#
#     # copied from tf2onnx project
#     # added extraction of signature from meta graph
#     """Load tensorflow graph from checkpoint."""
#     # make sure we start with clean default graph
#     tf_reset_default_graph()
#     # model_path = checkpoint/checkpoint.meta
#     with tf.device("/cpu:0"):
#         with tf_session() as sess:
#             saver = tf_import_meta_graph(model_path, clear_devices=True)
#             # restore from model_path minus the ".meta"
#             saver.restore(sess, model_path[:-5])
#             input_names = inputs_without_resource(sess, input_names)
#             frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
#             input_names = remove_redundant_inputs(frozen_graph, input_names)
#
#         tf_reset_default_graph()
#         with tf_session() as sess:
#             frozen_graph = tf_optimize(input_names, output_names, frozen_graph)
#     tf_reset_default_graph()
#     return frozen_graph, input_names, output_names


def to_savedmodel(graph_def, inputs: Dict[str, Any], outputs: Dict[str, Any], output_path):
    with tf.device("/cpu:0"):
        with tf_session() as sess:
            tf.import_graph_def(graph_def, name="")

        is_func = is_function(sess.graph)
        if not is_func:
            infer_shape(sess.graph, {})

        inputs = {name: sess.graph.get_tensor_by_name(spec.name) for name, spec in inputs.items()}
        outputs = {name: sess.graph.get_tensor_by_name(spec.name) for name, spec in outputs.items()}

        def _ensure_shape(tensors_dict, tensors_specs):
            for name, tensor in tensors_dict.items():
                if tensor.shape.rank is None:
                    tensor.set_shape(tensors_specs[name].shape)
            return tensors_dict

        inputs = _ensure_shape(inputs, inputs)
        outputs = _ensure_shape(outputs, outputs)

        print(inputs)
        print(outputs)

        tf.compat.v1.saved_model.simple_save(sess, output_path, inputs, outputs, legacy_init_op=None)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--checkpoint", help="foo help", required=True)
    parser.add_argument("--output", nargs="*", help="foo help", required=True)
    parser.add_argument("--inputs", help="foo help", required=True)
    parser.add_argument("--outputs", help="foo help", required=True)
    args = parser.parse_args()

    meta_graph_def = meta_graph.read_meta_graph_file(args.checkpoint)
    graph_util.remove_training_nodes(meta_graph_def.graph_def, args.outputs)

    frozen_graph, input_names, output_names = from_checkpoint(args.checkpoint, None, None)
    inputs = {}
    outputs = {}
    to_savedmodel(frozen_graph, inputs, outputs, args.output)


if __name__ == "__main__":
    main()
