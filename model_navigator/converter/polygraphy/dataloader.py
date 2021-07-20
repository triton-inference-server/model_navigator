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
#
from collections import OrderedDict

import numpy as np
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc

try:
    from polygraphy import constants
except ImportError:
    from polygraphy.common import constants  # to be deprecated in 0.32.0


class DataLoader:
    """
    Generates synthetic input data.
    """

    def __init__(
        self, seed=None, iterations=None, input_metadata=None, int_range=None, float_range=None, val_range=None
    ):
        """
        Args:
            seed (int):
                    The seed to use when generating random inputs.
                    Defaults to ``util.constants.DEFAULT_SEED``.
            iterations (int):
                    The number of iterations for which to supply data.
                    Defaults to 1.
            input_metadata (TensorMetadata):
                    A mapping of input names to their corresponding shapes and data types.
                    This will be used to determine what shapes to supply for inputs with dynamic shape, as
                    well as to set the data type of the generated inputs.
                    If either dtype or shape are None, then the value will be automatically determined.
                    For input shape tensors, i.e. inputs whose *value* describes a shape in the model, the
                    provided shape will be used to populate the values of the inputs, rather than to determine
                    their shape.
            int_range (Tuple[int]):
                    [DEPRECATED - Use val_range instead]
                    A tuple containing exactly 2 integers, indicating the minimum and maximum integer values (inclusive)
                    the data loader should generate. If either value in the tuple is None, the default will be used
                    for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
            float_range (Tuple[float]):
                    [DEPRECATED - Use val_range instead]
                    A tuple containing exactly 2 floats, indicating the minimum and maximum float values (inclusive)
                    the data loader should generate. If either value in the tuple is None, the default will be used
                    for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
            val_range (Union[Tuple[number], Dict[str, Tuple[number]]]):
                    A tuple containing exactly 2 numbers, indicating the minimum and maximum values (inclusive)
                    the data loader should generate.
                    If either value in the tuple is None, the default will be used for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
                    This can be specified on a per-input basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default range for inputs not explicitly listed.
        """

        def default_tuple(tup, default):
            if tup is None or not isinstance(tup, tuple) and not isinstance(tup, list):
                return default
            new_tup = []
            for elem, default_elem in zip(tup, default):
                new_tup.append(misc.default_value(elem, default_elem))
            return tuple(new_tup)

        self.seed = misc.default_value(seed, constants.DEFAULT_SEED)
        self.iterations = misc.default_value(iterations, 1)
        self.user_input_metadata = misc.default_value(input_metadata, {})

        self.int_range_set = int_range is not None
        self.int_range = default_tuple(int_range, (1, 25))

        self.float_range_set = float_range is not None
        self.float_range = default_tuple(float_range, (-1.0, 1.0))

        self.input_metadata = None
        self.val_range = misc.default_value(val_range, default_tuple(val_range, (0.0, 1.0)))

        if self.user_input_metadata:
            G_LOGGER.info(
                "Will generate inference input data according to provided TensorMetadata: {}".format(
                    self.user_input_metadata
                )
            )

    def _get_range(self, name, cast_type):
        if cast_type == int and self.int_range_set:
            return self.int_range
        elif cast_type == float and self.float_range_set:
            return self.float_range

        tup = None
        if isinstance(self.val_range, tuple):
            tup = self.val_range
        elif name in self.val_range:
            tup = self.val_range[name]
        elif "" in self.val_range:
            tup = self.val_range[""]

        if tup is None:
            G_LOGGER.critical(
                "Input tensor: {:} | Could not find value range."
                "\nNote: Provided value range was: {:}".format(name, self.val_range)
            )
        return tuple(cast_type(val) for val in tup)

    def __getitem__(self, index):  # noqa: C901
        """
        Randomly generates input data.

        May update the DataLoader's `input_metadata` attribute.

        Args:
            index (int):
                    Since this class behaves like an iterable, it takes an index parameter.
                    Generated data is guaranteed to be the same for the same index.

        Returns:
            OrderedDict[str, numpy.ndarray]: A mapping of input names to input numpy buffers.
        """
        if index >= self.iterations:
            raise IndexError()

        G_LOGGER.verbose(f"Generating data using numpy seed: {self.seed + index}")
        rng = np.random.RandomState(self.seed + index)

        def get_static_shape(name, shape):
            static_shape = shape
            if misc.is_shape_dynamic(shape):
                static_shape = misc.override_dynamic_shape(shape)
                if static_shape != shape and name not in self.user_input_metadata:
                    if not misc.is_valid_shape_override(static_shape, shape):
                        G_LOGGER.critical(
                            "Input tensor: {:} | Cannot override original shape: {:} to {:}".format(
                                name, shape, static_shape
                            )
                        )
                    G_LOGGER.warning(
                        "Input tensor: {:} | Will generate data of shape: {:} (tensor shape is: {:}).\n"
                        "If this is incorrect, please set input_metadata "
                        "or provide a custom data loader.".format(name, static_shape, shape),
                        mode=LogMode.ONCE,
                    )
            return static_shape

        # Whether the user provided the values for a shape tensor input,
        # rather than the shape of the input.
        # If the shape is 1D, and has a value equal to the rank of the provided default shape, it is
        # likely to be a shape tensor, and so its value, not shape, should be overriden.
        def is_shape_tensor(name, dtype):
            if name not in self.input_metadata or name not in self.user_input_metadata:
                return False

            _, shape = self.input_metadata[name]
            is_shape = np.issubdtype(dtype, np.integer) and (not misc.is_shape_dynamic(shape)) and (len(shape) == 1)

            user_shape = self.user_input_metadata[name].shape
            is_shape &= len(user_shape) == shape[0]
            is_shape &= not misc.is_shape_dynamic(user_shape)  # Shape of shape cannot be dynamic.
            return is_shape

        def generate_buffer(name, dtype, shape):
            if is_shape_tensor(name, dtype):
                buffer = np.array(shape, dtype=dtype)
                G_LOGGER.info(
                    "Assuming {:} is a shape tensor. Setting input values to: {:}. If this is not correct, "
                    "please set it correctly in 'input_metadata' or by providing --input-shapes".format(name, buffer),
                    mode=LogMode.ONCE,
                )
            elif np.issubdtype(dtype, np.integer):
                imin, imax = self._get_range(name, cast_type=int)
                # high is 1 greater than the max int drawn
                buffer = rng.randint(low=imin, high=imax + 1, size=shape, dtype=dtype)
            elif np.issubdtype(dtype, np.bool_):
                buffer = rng.randint(low=0, high=2, size=shape).astype(dtype)
            else:
                fmin, fmax = self._get_range(name, cast_type=float)
                buffer = (rng.random_sample(size=shape) * (fmax - fmin) + fmin).astype(dtype)

            buffer = np.array(buffer)  # To handle scalars, since the above functions return a float if shape is ().
            return buffer

        if self.input_metadata is None and self.user_input_metadata is not None:
            self.input_metadata = self.user_input_metadata

        buffers = OrderedDict()
        for name, (dtype, shape) in self.input_metadata.items():
            if name in self.user_input_metadata:
                user_dtype, user_shape = self.user_input_metadata[name]

                dtype = misc.default_value(user_dtype, dtype)

                is_valid_shape_override = user_shape is not None and misc.is_valid_shape_override(user_shape, shape)
                if not is_valid_shape_override and not is_shape_tensor(name, dtype):
                    G_LOGGER.warning(
                        "Input tensor: {:} | Cannot use provided custom shape: {:} "
                        "to override: {:}".format(name, user_shape, shape),
                        mode=LogMode.ONCE,
                    )
                else:
                    shape = misc.default_value(user_shape, shape)

            static_shape = get_static_shape(name, shape)
            buffers[name] = generate_buffer(name, dtype, shape=static_shape)

        # Warn about unused metadata
        for name in self.user_input_metadata.keys():
            if name not in self.input_metadata:
                msg = "Input tensor: {:} | Metadata was provided, but the input does not exist in one or more runners.".format(
                    name
                )
                close_match = misc.find_in_dict(name, self.input_metadata)
                if close_match:
                    msg += f"\nMaybe you meant to set: {close_match}"
                G_LOGGER.warning(msg)

        return buffers
