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
from typing import Dict

# pytype: disable=import-error
import tensorflow as tf
from tf2onnx import shape_inference, tf_loader

from model_navigator.tensor import TensorSpec

# pytype: enable=import-error


LOGGER = logging.getLogger(__name__)


def to_savedmodel(graph_def, inputs: Dict[str, TensorSpec], outputs: Dict[str, TensorSpec], output_path):
    with tf.device("/cpu:0"):
        with tf_loader.tf_session() as sess:
            tf.import_graph_def(graph_def, name="")

            is_func = tf_loader.is_function(sess.graph)
            if not is_func:
                shape_inference.infer_shape(sess.graph, {})

            input_tensors = {name: sess.graph.get_tensor_by_name(spec.name) for name, spec in inputs.items()}
            output_tensors = {name: sess.graph.get_tensor_by_name(spec.name) for name, spec in outputs.items()}

            def _ensure_shape(tensors_dict, tensors_specs):
                for name, tensor in tensors_dict.items():
                    if tensor.shape.rank is None:
                        tensor.set_shape(tensors_specs[name].shape)
                return tensors_dict

            input_tensors = _ensure_shape(input_tensors, inputs)
            output_tensors = _ensure_shape(output_tensors, outputs)

            LOGGER.info(input_tensors)
            LOGGER.info(output_tensors)

            tf.compat.v1.saved_model.simple_save(sess, output_path, input_tensors, output_tensors, legacy_init_op=None)
