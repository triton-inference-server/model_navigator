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

from typing import Dict, List

import fire
import numpy
import tensorflow as tf  # pytype: disable=import-error
from jax.experimental import jax2tf  # pytype: disable=import-error


def get_model():
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def get_model_params():
    raise NotImplementedError("Please implement the get_model_params() function to reproduce the export error.")


def export(
    exported_model_path: str,
    jit_compile: bool,
    enable_xla: bool,
    input_metadata: List[Dict],
):
    model = get_model()
    model_params = get_model_params()

    polymorphic_shapes = []
    tensor_spec = []
    for m in input_metadata:
        p_shape = []
        p_shape_idx = 0
        tf_shape = []
        for s in m["shape"]:
            if s == -1:
                p_shape.append(f"dim_{p_shape_idx}")
                p_shape_idx += 1
                tf_shape.append(None)
            else:
                p_shape.append(str(s))
                tf_shape.append(s)
        tensor_spec.append(tf.TensorSpec(name=m["name"], shape=tf_shape, dtype=numpy.dtype(m["dtype"])))
        polymorphic_shapes.append(f"({','.join(p_shape)})")
    params_vars = tf.nest.map_structure(tf.Variable, model_params)
    tf_function = jax2tf.convert(
        model, polymorphic_shapes=polymorphic_shapes, enable_xla=enable_xla, with_gradient=True
    )

    def tf_function_wrapper(*args):
        return tf_function(*args, params=params_vars)

    tf_module = tf.Module()
    tf_module._variables = tf.nest.flatten(params_vars)
    tf_module.f = tf.function(
        tf_function_wrapper,
        autograph=False,
        jit_compile=jit_compile,
        input_signature=tensor_spec,
    )

    signatures = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf_module.f.get_concrete_function(*tensor_spec)}

    tf.saved_model.save(
        tf_module,
        exported_model_path,
        signatures=signatures,
    )


if __name__ == "__main__":
    fire.Fire(export)
