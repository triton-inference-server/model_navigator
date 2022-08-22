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

import numpy as np

from model_navigator.exceptions import ModelNavigatorConverterException
from model_navigator.model import ModelSignatureConfig
from model_navigator.tensor import TensorSpec

LOGGER = logging.getLogger(__name__)


def get_tf_signature(model_path):
    import tensorflow as tf  # pytype: disable=import-error

    def conv_shape(shape):
        xl = shape.as_list()
        for i, x in enumerate(xl):
            if not x:
                xl[i] = -1
        return tuple(xl)

    model = tf.saved_model.load(model_path.as_posix())

    try:
        concrete_func = model.signatures["serving_default"]

        # inspired by https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/saved_model_cli.py#L205
        if concrete_func.structured_input_signature:
            input_args, input_kwargs = concrete_func.structured_input_signature
            input_names = list(input_kwargs)
            assert not input_args, f"Unsupported positional args in concrete function signature: args={input_args}"
            inputs = {
                name: TensorSpec(sig.name, conv_shape(sig.shape), np.dtype(sig.dtype.as_numpy_dtype))
                for name, sig in input_kwargs.items()
            }
        elif concrete_func._arg_keywords:  # pylint: disable=protected-access
            # For pure ConcreteFunctions we might have nothing better than _arg_keywords.
            assert concrete_func._num_positional_args in [0, 1]
            input_names = concrete_func._arg_keywords

            # TODO: Is this needed at all? Is this even correct?
            input_tensors = [tensor for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
            inputs = {
                name: TensorSpec(tensor.name, conv_shape(tensor.shape), np.dtype(tensor.dtype.as_numpy_dtype))
                for name, tensor in zip(input_names, input_tensors)
            }
        else:
            raise ModelNavigatorConverterException("Unsupported `concrete_func` when loading model signature.")

        outputs = {
            tensor.name: TensorSpec(tensor.name, conv_shape(tensor.shape), np.dtype(tensor.dtype.as_numpy_dtype))
            for tensor in concrete_func.outputs
        }

        return ModelSignatureConfig(inputs=inputs, outputs=outputs)
    except Exception as e:
        LOGGER.error(e)
        raise


def obtain_inputs(concrete_func):
    """Get input names for a TF2 ConcreteFunc"""
    import tensorflow as tf  # pytype: disable=import-error

    # inspired by https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/saved_model_cli.py#L205
    if concrete_func.structured_input_signature:
        input_args, input_kwargs = concrete_func.structured_input_signature
        input_names = list(input_kwargs)
        assert (
            not input_args
        ), f"Not supported args in concrete function signature args={input_args}, kwargs={input_kwargs}"
    elif concrete_func._arg_keywords:  # pylint: disable=protected-access
        # For pure ConcreteFunctions we might have nothing better than _arg_keywords.
        assert concrete_func._num_positional_args in [0, 1]
        input_names = concrete_func._arg_keywords

    input_tensors = [tensor for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
    inputs = {name: tensor.name for name, tensor in zip(input_names, input_tensors)}
    return inputs
