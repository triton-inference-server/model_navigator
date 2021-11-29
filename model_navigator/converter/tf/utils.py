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

# pytype: disable=import-error
import tensorflow as tf
# pytype: enable=import-error

from model_navigator.exceptions import ModelNavigatorProfileException

LOGGER = logging.getLogger(__name__)


def obtain_inputs(concrete_func):
    """ Get input names for a TF2 ConcreteFunc """
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


def get_default_profile(concrete_func, max_batch_size: int):
    shapes = {
        'min': {},
        'max': {},
    }
    if max_batch_size == 0:
        return None
    if max_batch_size < 0:
        raise ModelNavigatorProfileException(
            "Cannot construct default dataset profile: max_batch_size negative"
        )
    for inp in concrete_func.inputs:
        min_shape = inp.shape.as_list()
        min_shape[0] = 1
        max_shape = inp.shape.as_list()
        max_shape[0] = max_batch_size
        for x in max_shape:
            if not x:
                raise ModelNavigatorProfileException(
                    "Cannot construct default dataset profile: "
                    f"too many dynamic axes in the model input {inp.name}: {inp.shape.as_list()}. "
                    "Please provide a full dataset profile instead of max_batch_size."
                )
        shapes['min'][inp.name] = min_shape
        shapes['max'][inp.name] = max_shape
    return shapes


def generate_inputs(concrete_func, shapes, value_ranges=None):
    """ Generate random input data for `concrete_func`.
    """
    func_inputs = {v: k for k, v in obtain_inputs(concrete_func).items()}
    if not value_ranges:
        value_ranges = {}
        for inp in func_inputs.values():
            value_ranges[inp] = (0.0, 1.0)
        LOGGER.info(f"Value ranges not provided, using default: {value_ranges}.")

    def generate_sample(shapes):
        profile_inputs = sorted(shapes.keys())
        if sorted(func_inputs.values()) != profile_inputs:
            raise ModelNavigatorProfileException(
                "Dataset profile does not match model inputs. "
                f"ConcreteFunction inputs: {func_inputs}; "
                f"profile inputs: {profile_inputs}."
            )
        sample = []
        for inp in concrete_func.inputs:
            sample_inp_shape = []
            for x, px in zip(inp.shape, shapes[func_inputs[inp.name]]):
                if x and x != px:
                    raise ModelNavigatorProfileException(
                        "Dataset profile does not match model inputs. "
                        f"For input {inp.name}, shape is {inp.shape} "
                        f"but profile specifies: {shapes[func_inputs[inp.name]]}."
                    )
                if not x:
                    sample_inp_shape.append(px)
                else:
                    sample_inp_shape.append(x)
            ranges = value_ranges[func_inputs[inp.name]]
            sample.append(tf.random.uniform(tuple(sample_inp_shape),
                                            minval=ranges[0],
                                            maxval=ranges[1]))
        return sample

    inputs = [
        generate_sample(shape) for shape in shapes.values()
    ]

    def _gen():
        for inp in inputs:
            yield tuple(inp)
    return _gen
