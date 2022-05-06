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

# pytype: disable=import-error
import tensorflow as tf

# pytype: enable=import-error


LOGGER = logging.getLogger(__name__)


def obtain_inputs(concrete_func):
    """Get input names for a TF2 ConcreteFunc"""
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
