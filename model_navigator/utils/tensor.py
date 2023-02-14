# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Utilities for working with tensors in different environments."""
import abc
import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

# pytype: disable=annotation-type-mismatch
# pytype: disable=wrong-keyword-args
# pytype: disable=name-error


@dataclasses.dataclass
class TensorSpec:
    """Stores specification of single tensor. This includes name, shape and dtype.

    Shape should be a tuple of positive integers or -1. -1 means dynamic dimension. dtype should be np.dtype.
    Both arguments should be passed as keyword arguments.

    Example of use:
        tensor_sepc = TensorSpec(
        name="images",
        dtype=np.dtype("uint8"),
        shape=(-1, 28, 28, 1)
    )

    Raises TypeError if arguments validation fails due to wrong type or value.
    """

    name: str
    shape: Tuple
    dtype: Optional[np.dtype] = dataclasses.field(default=None)
    optional: Optional[bool] = False

    def __post_init__(self):
        """Validate the configuration for early error handling."""

        def _expect_type(name, value, expected_types, optional=False):
            if not (isinstance(value, expected_types) or (value is None and optional)):
                raise TypeError(f"{name} should be {expected_types}, but got {type(value)}")

        def _is_dim_correct(dim):
            # int equal to -1 or positive number
            return isinstance(dim, int) and (dim == -1 or dim > 0)

        _expect_type("name", self.name, str)
        _expect_type("shape", self.shape, tuple)
        _expect_type("dtype", self.dtype, np.dtype, optional=True)
        _expect_type("optional", self.optional, bool, optional=True)
        if not all([_is_dim_correct(dim) for dim in self.shape]):
            raise TypeError(f"Shape items should be integers equal to -1 or positive numbers. Got {self.shape}")

    def astype(self, dtype: Union[np.dtype, Type[np.dtype]]) -> "TensorSpec":
        """Change the TensorSpec dtype."""
        tensor = TensorSpec(name=self.name, shape=self.shape, dtype=np.dtype(dtype), optional=self.optional)
        return tensor

    def is_dtype_compatible(self, tensor: np.ndarray) -> bool:
        """Check if `tensor` has dtype compatible with `self`. E.g. dtype of `tensor` is subtype of `self`."""
        return np.issubdtype(tensor.dtype, self.dtype)

    def is_shape_compatible(self, tensor: np.ndarray) -> bool:
        """Check if `tensor` has shape compatible with `self`. E.g. `tensor` has shape (3, 1) and `self` has (-1, 1)."""
        if len(self.shape) != len(tensor.shape):
            return False
        for self_dim, spec_dim in zip(self.shape, tensor.shape):
            if self_dim != spec_dim and self_dim != -1:
                return False
        return True


class TensorUtils(abc.ABC):
    """Abstract class for utils of different implementations of tensor data."""

    @staticmethod
    def for_data(data):
        """Return correct utils for provided data."""
        if isinstance(data, dict):
            types = {type(value) for value in data.values()}
            if len(types) != 1:
                raise ValueError("Could not handle different data types in dict")
            data = list(data.values())[0]
        elif isinstance(data, (tuple, list)):
            types = [type(value) for value in data]
            if len(types) != 1:
                raise ValueError("Could not handle different data types in tuple/list")
            data = data[0]

        data_type = type(data)
        framework = data_type.__module__.split(".")[0]  # take first part of package name
        return {
            "torch": PyTorchTensorUtils,
            "tensorflow": TensorFlowTensorUtils,
            "numpy": NumpyTensorUtils,
            "builtins": BuiltinsTensorUtils,  # ex. list, tuples
        }[framework]

    @staticmethod
    @abc.abstractmethod
    def eq(a, b):
        """Comparison of two tensors."""
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def to_numpy(a):
        """Cast tensor to numpy format."""
        raise NotImplementedError()


class PyTorchTensorUtils(TensorUtils):
    """Utils for PyTorch tensors."""

    @staticmethod
    def eq(a, b):
        """Comparison of two tensors."""
        import torch  # pytype: disable=import-error

        return a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and torch.all(torch.eq(a, b))

    @staticmethod
    def to_numpy(a):
        """Cast tensor to numpy format."""
        return a.cpu().detach().numpy()


class TensorFlowTensorUtils(TensorUtils):
    """Utils for TensorFlow tensors."""

    @staticmethod
    def eq(a, b):
        """Comparison of two tensors."""
        import tensorflow as tf  # pytype: disable=import-error

        return a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and tf.reduce_all(a == b)

    @staticmethod
    def to_numpy(a):
        """Cast tensor to numpy format."""
        return a.numpy()


class NumpyTensorUtils(TensorUtils):
    """Utils for numpy tensors."""

    @staticmethod
    def eq(a, b):
        """Comparison of two tensors."""
        # np.array_equal checks shape and content
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)

    @staticmethod
    def to_numpy(a):
        """Cast tensor to numpy format."""
        return a


class BuiltinsTensorUtils(TensorUtils):
    """Utils for Python builtins."""

    @staticmethod
    def eq(a, b):
        """Comparison of two tensors."""
        return a == b

    @staticmethod
    def to_numpy(a):
        """Cast tensor to numpy format."""
        return np.array(a)


class TensorMetadata(Dict[str, TensorSpec]):
    """Metadata for inputs/outputs tensors."""

    def add(self, name: str, shape: Sequence[int], dtype: Union[np.dtype, Type[np.dtype]]) -> None:
        """Add new item to metadata.

        Args:
            name: Name of tensor
            shape: Shape of tensor
            dtype: Type of tensor data
        """
        self[name] = TensorSpec(name, tuple(shape), np.dtype(dtype))

    @classmethod
    def from_json(cls, data: List[Dict]) -> "TensorMetadata":
        """Create object JSON data format.

        Args:
            data: A list with tensors data

        Returns:
            List converted to TensorMetadata object
        """
        tensor_metadata = cls()
        for value in data:
            tensor_metadata.add(value["name"], value["shape"], value["dtype"])
        return tensor_metadata

    def to_json(self) -> List[Dict]:
        """Convert object to JSON serializable format.

        Returns:
            List of dictionaries with tensors data.
        """
        data = []
        for spec in self.values():
            data.append(self._parse_tensorspec(spec))
        return data

    @property
    def dynamic_axes(self) -> Dict:
        """Return information for shapes with dynamic axies per each tensor."""
        dynamic_axes = defaultdict(list)
        for name, tensor_spec in self.items():
            for ax, d in enumerate(tensor_spec.shape):
                if d == -1:
                    dynamic_axes[name].append(ax)
        return dynamic_axes

    @staticmethod
    def _parse_tensorspec(spec: TensorSpec):
        return {"name": spec.name, "shape": spec.shape, "dtype": str(spec.dtype)}
