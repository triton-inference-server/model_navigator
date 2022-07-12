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
import abc
import dataclasses
from typing import Any, Dict, Optional, Tuple

import numpy as np

# pytype: disable=annotation-type-mismatch
# pytype: disable=wrong-keyword-args
# pytype: disable=name-error


@dataclasses.dataclass
class TensorSpec:
    """Stores specification of single tensor. This includes name, shape and dtype.

    Shape should be a tuple of positive integers or -1. -1 means dynamic dimension. dtype should be np.dtype.
    Both arguments should be passed as keyword arguments.

    >>> TensorSpec(name="images", dtype=np.dtype("uint8"), shape=(-1, 28, 28, 1))

    Class contains also helper classmethod converting 3rd party known formats to TensorSpec.

    >>> from model_navigator.triton import TritonClient
    >>> client = TritonClient("grpc://127.0.0.1:8001")
    >>> model_metadata = client.get_model_metadata("ResNet50", "1")
    >>> spec = [TensorSpec.from_triton_tensor_metadata(metadata) for metadata in model_metadata.inputs]

    >>> import onnx
    >>> from polygraphy.backend.onnx import OnnxFromPath
    >>> from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata
    >>> model: onnx.ModelProto = OnnxFromPath("/tmp/model.onnx")()
    >>> inputs = get_input_metadata(model.graph)
    >>> [TensorSpec.from_polygraphy_metadata_tuple(name=name, metadata=metadata) for name, metadata in inputs.items()]

    Raises TypeError if arguments validation fails due to wrong type or value.
    """

    name: str
    shape: Tuple
    dtype: Optional[np.dtype] = dataclasses.field(default=None)
    optional: Optional[bool] = False

    def __post_init__(self):
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

    def is_dynamic(self):
        """Check if tensor is dynamic - if any of dimension have -1 in shape. Except fist axis which is batch size."""
        return any([dim == -1 for dim in self.shape[1:]])

    def astype(self, dtype) -> "TensorSpec":
        """Change the TensorSpec dtype"""
        tensor = TensorSpec(name=self.name, shape=self.shape, dtype=np.dtype(dtype), optional=self.optional)
        return tensor

    @classmethod
    def from_triton_tensor_metadata(cls, tensor_metadata: Dict[str, Any]):
        """Wraps Triton TensorMetadata into TensorSpec

        Args:
            tensor_metadata: TensorMetadata object to wrap

        Returns:
            TensorSpec based on data from provided TensorMetadata.
        """
        from model_navigator.triton import client_utils

        return cls(
            name=tensor_metadata["name"],
            shape=tuple(int(s) for s in tensor_metadata["shape"]),
            dtype=np.dtype(client_utils.triton_to_np_dtype(tensor_metadata["datatype"])),
            optional=bool(tensor_metadata.get("optional", False)),
        )

    @classmethod
    def from_polygraphy_metadata_tuple(cls, name: str, metadata: "MetadataTuple"):  # noqa: F821
        """Wraps Polygraphy MetadataTuple into TensorSpec.

        Args:
            name: name of the tensor
            metadata: MetadataTuple containing shape and dtype

        Returns:
            TensorSpec based on data from provided MetadataTuple.
        """
        return cls(
            name=name,
            shape=tuple(dim if isinstance(dim, int) else -1 for dim in metadata.shape),
            dtype=metadata.dtype,
            optional=False,
        )


class TensorUtils(abc.ABC):
    @staticmethod
    def for_data(data):
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
            "torch": PyTTensorUtils,
            "tensorflow": TFTensorUtils,
            "numpy": NPTensorUtils,
            "builtins": BuiltinsTensorUtils,  # ex. list, tuples
        }[framework]

    @staticmethod
    @abc.abstractmethod
    def eq(a, b):
        pass

    @staticmethod
    @abc.abstractmethod
    def to_numpy(a):
        pass


class PyTTensorUtils(TensorUtils):
    @staticmethod
    def eq(a, b):
        # pytype: disable=import-error
        import torch

        # pytype: enable=import-error

        return a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and torch.all(torch.eq(a, b))

    @staticmethod
    def to_numpy(a):
        return a.cpu().detach().numpy()


class TFTensorUtils(TensorUtils):
    @staticmethod
    def eq(a, b):
        # pytype: disable=import-error
        import tensorflow as tf

        # pytype: enable=import-error

        return a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and tf.reduce_all(a == b)

    @staticmethod
    def to_numpy(a):
        return a.numpy()


class NPTensorUtils(TensorUtils):
    @staticmethod
    def eq(a, b):
        # np.array_equal checks shape and content
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)

    @staticmethod
    def to_numpy(a):
        return a


class BuiltinsTensorUtils(TensorUtils):
    @staticmethod
    def eq(a, b):
        return a == b

    @staticmethod
    def to_numpy(a):
        return np.array(a)
