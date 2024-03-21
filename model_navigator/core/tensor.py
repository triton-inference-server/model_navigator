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
import itertools

# import json
import uuid  # TODO find better solution
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from model_navigator.api.config import TensorType
from model_navigator.frameworks import Framework, is_jax_available, is_tf_available, is_torch_available
from model_navigator.frameworks.tensorrt.cuda import DeviceView
from model_navigator.utils import common, module
from model_navigator.utils.common import PYTHON_PRIMITIVE_TYPES

torch = module.lazy_import("torch")
tf = module.lazy_import("tensorflow")
jax = module.lazy_import("jax")
jaxlib = module.lazy_import("jaxlib")

# pytype: disable=annotation-type-mismatch
# pytype: disable=wrong-keyword-args
# pytype: disable=name-error


@dataclasses.dataclass
class TensorSpec:
    """Stores specification of single tensor. This includes name, shape and dtype.

    Shape should be a tuple of positive integers or -1. -1 means dynamic dimension. dtype should be np.dtype.
    Both arguments should be passed as keyword arguments.

    Example of use:
        tensor_spec = TensorSpec(
        name="images",
        dtype=np.dtype("uint8"),
        shape=(-1, 28, 28, 1)
    )

    Raises TypeError if arguments validation fails due to wrong type or value.
    """

    name: str
    shape: Tuple
    dtype: Optional[Union[np.dtype, "torch.dtype"]] = dataclasses.field(default=None)
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
        if is_torch_available():
            expected_types = (np.dtype, torch.dtype)
        else:
            expected_types = (np.dtype,)
        _expect_type("dtype", self.dtype, expected_types, optional=True)
        _expect_type("optional", self.optional, bool, optional=True)
        if not all(_is_dim_correct(dim) for dim in self.shape):
            raise TypeError(f"Shape items should be integers equal to -1 or positive numbers. Got {self.shape}")

    def astype(self, dtype: Union[np.dtype, Type[np.dtype]]) -> "TensorSpec":
        """Change the TensorSpec dtype."""
        tensor = TensorSpec(name=self.name, shape=self.shape, dtype=np.dtype(dtype), optional=self.optional)
        return tensor

    def is_dtype_compatible(self, spec: "TensorSpec") -> bool:
        """Check if `tensor` has dtype compatible with `self`. E.g. dtype of `tensor` is subtype of `self`."""
        return np.issubdtype(spec.dtype, self.dtype)

    def is_shape_compatible(self, spec: "TensorSpec") -> bool:
        """Check if `tensor` has shape compatible with `self`. E.g. `tensor` has shape (3, 1) and `self` has (-1, 1)."""
        if len(self.shape) != len(spec.shape):
            return False
        for self_dim, spec_dim in zip(self.shape, spec.shape):
            if self_dim != spec_dim and self_dim != -1:
                return False
        return True

    @classmethod
    def from_numpy_tensor(cls, tensor: np.ndarray, name: str) -> "TensorSpec":
        """Create TensorSpec from numpy array."""
        return cls(name=name, shape=tensor.shape, dtype=np.dtype(tensor.dtype))

    @classmethod
    def from_torch_tensor(cls, tensor: "torch.Tensor", name: str) -> "TensorSpec":
        """Create TensorSpec from torch tensor."""
        return cls(name=name, shape=tensor.shape, dtype=np.dtype(common.torch_to_numpy_dtype(tensor.dtype)))

    @classmethod
    def from_tensor(cls, tensor, name: str) -> "TensorSpec":
        """Create TensorSpec from tensor."""
        tensor_type = get_tensor_type(tensor)
        if tensor_type == TensorType.NUMPY:
            return cls.from_numpy_tensor(tensor, name)
        elif tensor_type == TensorType.TORCH:
            return cls.from_torch_tensor(tensor, name)
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")


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
        return a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and torch.all(torch.eq(a, b))

    @staticmethod
    def to_numpy(a):
        """Cast tensor to numpy format."""
        # TODO: remove bfloat16 special case once torch.bfloat16 is supported
        return a.cpu().detach().numpy() if a.dtype != torch.bfloat16 else a.to(torch.float32).cpu().detach().numpy()


class TensorFlowTensorUtils(TensorUtils):
    """Utils for TensorFlow tensors."""

    @staticmethod
    def eq(a, b):
        """Comparison of two tensors."""
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


def is_tensor(tensor: Any, tensor_type: TensorType) -> bool:
    """Validate if provided object is a valid tensor.

    Args:
        tensor: An object to validate
        tensor_type: A framework for which the object has to be tested

    Returns:
        True if object is a valid tensor, False otherwise
    """
    if tensor_type == TensorType.TORCH:
        return torch.is_tensor(tensor) or isinstance(tensor, (np.ndarray, DeviceView))
    elif tensor_type == TensorType.TENSORFLOW:
        return tf.is_tensor(tensor) or isinstance(tensor, np.ndarray)
    elif tensor_type == TensorType.JAX:
        return isinstance(tensor, (np.ndarray, jax.numpy.ndarray, jaxlib.xla_extension.ArrayImpl))
    else:
        return isinstance(tensor, np.ndarray)


class PyTreeMetadata:
    """Description of the metadata of a tree of tensors."""

    def __init__(self, metadata: Any, tensor_type: TensorType) -> None:
        """Create PyTreeMetadata from provided metadata."""
        self._metadata = metadata
        self.tensor_type = tensor_type

    def __str__(self) -> str:
        """Convert PyTree metadata to string."""
        return str(self._metadata)

    def __eq__(self, __value: object) -> bool:
        """Compare PyTree metadata."""
        if not isinstance(__value, type(self)):
            return False
        return self._metadata == __value._metadata

    def __hash__(self) -> int:
        """Compute hash of PyTree metadata."""
        return self._hash(self._metadata, 0)

    @classmethod
    def from_sample(
        cls, sample: Any, tensor_type: TensorType, names: Optional[Iterable[str]] = None, prefix: str = ""
    ) -> "PyTreeMetadata":
        """Create PyTreeMetadata from sample.

        Args:
            sample: A sample from which PyTreeMetadata will be created
            tensor_type: A type of tensors in the sample
            names: Names of tensors in the sample
            prefix: A prefix for names of tensors in the sample. Used only if names are not provided.

        Returns:
            PyTreeMetadata created from sample
        """
        if names is None:
            assert prefix != "", "Prefix must be provided if names are not provided"
            names = (f"{prefix}__{i}" for i in itertools.count(start=0, step=1))
        else:
            names = iter(names)
        metadata, _ = cls._from_sample(sample, tensor_type, names)
        return cls(metadata, tensor_type)

    def to_dict(self) -> Dict:
        """Convert PyTree metadata to string."""
        return {
            "metadata": self._metadata,
            "tensor_type": self.tensor_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PyTreeMetadata":
        """Create PyTreeMetadata from string."""
        return cls(cls._from_json(data["metadata"]), TensorType(data["tensor_type"]))

    def flatten_sample(self, sample: Any) -> Dict[str, Any]:
        """Flatten sample according to PyTree metadata.

        Returns flatten dictionary with keys corresponding to PyTree metadata.
        """
        flattened_sample = {}
        self._flatten_sample(sample, self._metadata, flattened_sample)
        return flattened_sample

    def unflatten_sample(self, sample: Dict[str, Any], wrap_input: bool = False) -> Any:
        """Unflatten sample according to PyTree metadata.

        Returns unflatten sample according to PyTree metadata.
        If wrap_input is True, then single tensor will be wrapped in tuple.
        """
        unflatten_sample = self._unflatten_sample(sample, self._metadata)
        if wrap_input and isinstance(self._metadata, (str, Mapping)):
            unflatten_sample = (unflatten_sample,)
        return unflatten_sample

    def is_compatible_with(self, sample: Any) -> bool:
        """Check if sample is compatible with PyTreeMetadata.

        Args:
            sample: A sample to check compatibility with PyTreeMetadata

        Returns:
            True if sample is compatible with PyTreeMetadata, False otherwise
        """
        return self._is_compatible_with(self._metadata, sample)

    def get_names_mapping(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        """Get mapping of PyTree metadata to names."""
        metadata = self._metadata
        if isinstance(metadata, (str, Mapping)):
            metadata = (metadata,)

        if isinstance(metadata[-1], Mapping):
            args, kwargs = metadata[:-1], metadata[-1]
        else:
            args, kwargs = metadata, {}

        args_mapping, kwargs_mapping = [], {}
        for arg in args:
            flattened = {}
            self._flatten_sample(arg, arg, flattened, include_constants=False)
            args_mapping.append(list(flattened.keys()))

        for key, arg in kwargs.items():
            flattened = {}
            self._flatten_sample(arg, arg, flattened, include_constants=False)
            kwargs_mapping[key] = list(flattened.keys())

        return args_mapping, kwargs_mapping

    def get_names(self) -> Sequence[str]:
        """Get names of tensors in PyTreeMetadata."""
        return list(self.flatten_sample(self._metadata).keys())

    def _is_compatible_with(self, metadata, sample):
        if isinstance(metadata, str):
            return is_tensor(sample, self.tensor_type)
        elif isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            if not isinstance(sample, PYTHON_PRIMITIVE_TYPES):
                return False
            return metadata == sample
        elif isinstance(metadata, Mapping):
            if not isinstance(sample, Mapping) or set(metadata) != set(sample):
                return False
            for key, item in sample.items():
                if not self._is_compatible_with(metadata[key], item):
                    return False
            return True
        elif isinstance(sample, Sequence):
            if not isinstance(metadata, Sequence) or len(metadata) != len(sample):
                return False
            for item, submetadata in zip(sample, metadata):
                if not self._is_compatible_with(submetadata, item):
                    return False
            return True
        else:
            raise TypeError(f"Unsupported type: {type(sample)}")

    @classmethod
    def _from_sample(cls, sample, tensor_type, names):
        """Create PyTreeMetadata from sample."""
        if is_tensor(sample, tensor_type):
            return next(names), names
        if isinstance(sample, PYTHON_PRIMITIVE_TYPES):
            return sample, names
        if isinstance(sample, Mapping):
            metadata = {}
            for key, item in sorted(sample.items()):
                submetadata, names = cls._from_sample(item, tensor_type, names)
                metadata[key] = submetadata
            return metadata, names
        if isinstance(sample, Sequence):
            metadata = []
            for item in sample:
                submetadata, names = cls._from_sample(item, tensor_type, names)
                metadata.append(submetadata)
            return tuple(metadata), names
        raise TypeError(f"Unsupported type: {type(sample)}")

    @classmethod
    def _from_json(cls, sample):
        """Create PyTreeMetadata from sample."""
        if isinstance(sample, str) or isinstance(sample, PYTHON_PRIMITIVE_TYPES):
            return sample
        if isinstance(sample, Mapping):
            metadata = {}
            for key, item in sorted(sample.items()):
                submetadata = cls._from_json(item)
                metadata[key] = submetadata
            return metadata
        if isinstance(sample, Sequence):
            metadata = []
            for item in sample:
                submetadata = cls._from_json(item)
                metadata.append(submetadata)
            return tuple(metadata)
        raise TypeError(f"Unsupported type: {type(sample)}")

    def _flatten_sample(self, sample, struct, flatten_sample, include_constants=False):
        if isinstance(struct, str):
            flatten_sample[struct] = sample
        elif isinstance(sample, PYTHON_PRIMITIVE_TYPES):
            if include_constants:
                flatten_sample[f"const_{uuid.uuid4()}"] = sample
        elif isinstance(sample, Mapping):
            assert isinstance(struct, Mapping)
            for key, item in sample.items():
                self._flatten_sample(item, struct[key], flatten_sample, include_constants=include_constants)
        elif isinstance(sample, Sequence):
            assert isinstance(struct, Sequence)
            i = 0
            for item in sample:
                self._flatten_sample(item, struct[i], flatten_sample, include_constants=include_constants)
                i += 1
        else:
            raise TypeError(f"Unsupported type: {type(sample)}")

    def _unflatten_sample(self, sample, struct):
        if isinstance(struct, str):
            return sample[struct]
        elif isinstance(struct, PYTHON_PRIMITIVE_TYPES):
            return struct
        elif isinstance(struct, Mapping):
            return {key: self._unflatten_sample(sample, item) for key, item in struct.items()}
        elif isinstance(struct, Sequence):
            return type(struct)(self._unflatten_sample(sample, item) for item in struct)
        else:
            raise TypeError(f"Unsupported struct: {struct}")

    def _hash(self, struct, hash_):
        if isinstance(struct, str) or isinstance(struct, PYTHON_PRIMITIVE_TYPES):
            return hash_ ^ hash(struct)
        elif isinstance(struct, Mapping):
            for key, value in struct.items():
                hash_ ^= hash(key)
                hash_ ^= self._hash(value, hash_)
            return hash_
        elif isinstance(struct, Sequence):
            for value in struct:
                hash_ ^= self._hash(value, hash_)
            return hash_
        else:
            raise TypeError(f"Unsupported struct: {struct}")


class TensorMetadata(Dict[str, TensorSpec]):
    """Metadata for inputs/outputs tensors."""

    def __init__(self, *args, pytree_metadata: Optional[PyTreeMetadata] = None, is_legacy: bool = False, **kwargs):
        """Create TensorMetadata object.

        Args:
            args: Arguments for dict
            pytree_metadata: Description of the PyTree metadata of the tensors
            is_legacy: If True use legacy flatten_sample. Should be set to True for packages older than v0.2.4
            kwargs: Keyword arguments for dict
        """
        super().__init__(*args, **kwargs)
        if pytree_metadata is None:
            pytree_metadata = PyTreeMetadata(None, TensorType.NUMPY)
        self._pytree_metadata = pytree_metadata
        self.is_legacy = is_legacy

    def add(
        self,
        name: str,
        shape: Sequence[int],
        dtype: Union[np.dtype, Type[np.dtype], "torch.dtype", "Type[torch.dtype]"],
    ) -> "TensorMetadata":
        """Add new item to metadata.

        Args:
            name: Name of tensor
            shape: Shape of tensor
            dtype: Type of tensor data
        """
        # TODO: remove bfloat16 special case once torch.bfloat16 is supported
        if dtype == "torch.bfloat16":
            dtype = torch.bfloat16
        elif not (is_torch_available() and isinstance(dtype, torch.dtype)):
            dtype = np.dtype(dtype)
        self[name] = TensorSpec(name, tuple(shape), dtype)
        return self

    @property
    def pytree_metadata(self) -> PyTreeMetadata:
        """Return PyTree metadata."""
        return self._pytree_metadata

    @classmethod
    def from_json(cls, data: Dict) -> "TensorMetadata":
        """Create object JSON data format.

        Args:
            data: A list with tensors data

        Returns:
            List converted to TensorMetadata object
        """
        tensor_metadata = cls(
            pytree_metadata=PyTreeMetadata.from_dict(data["pytree_metadata"]), is_legacy=data.get("is_legacy", False)
        )
        for value in data["metadata"]:
            tensor_metadata.add(value["name"], value["shape"], value["dtype"])
        return tensor_metadata

    def to_json(self) -> Dict[str, Any]:
        """Convert object to JSON serializable format.

        Returns:
            List of dictionaries with tensors data.
        """
        metadata = []
        for spec in self.values():
            metadata.append(self._parse_tensorspec(spec))
        return {
            "metadata": metadata,
            "pytree_metadata": self.pytree_metadata.to_dict(),
            "is_legacy": self.is_legacy,
        }

    @property
    def dynamic_axes(self) -> Dict:
        """Return information for shapes with dynamic axes per each tensor."""
        dynamic_axes = defaultdict(list)
        for name, tensor_spec in self.items():
            for ax, d in enumerate(tensor_spec.shape):
                if d == -1:
                    dynamic_axes[name].append(ax)
        return dynamic_axes

    @staticmethod
    def _parse_tensorspec(spec: TensorSpec):
        return {"name": spec.name, "shape": spec.shape, "dtype": str(spec.dtype)}

    def flatten_sample(self, sample: Any) -> Dict[str, Any]:
        """Flatten sample according to PyTree metadata.

        Returns flatten dictionary with keys corresponding to PyTree metadata.
        """
        if not self.is_legacy:
            return self.pytree_metadata.flatten_sample(sample)
        else:
            return self._legacy_flatten_sample(sample)

    def unflatten_sample(self, sample: Dict[str, Any], wrap_input: bool = False) -> Any:
        """Unflatten sample according to PyTree metadata.

        Returns unflatten sample according to PyTree metadata.
        If wrap_input is True, then single tensor will be wrapped in tuple.
        """
        return self.pytree_metadata.unflatten_sample(sample, wrap_input=wrap_input)

    def _legacy_flatten_sample(self, sample: Any) -> Dict[str, Any]:
        """Flatten sample without PyTree metadata."""
        if is_tensor(sample, self.pytree_metadata.tensor_type):
            sample = (sample,)
        if isinstance(sample, Mapping):
            sample = sample.values()

        flattened_sample = dict(zip(self.keys(), sample))

        return flattened_sample


def get_tensor_type(tensor: Any) -> TensorType:
    """Get tensor type from tensor."""
    if isinstance(tensor, np.ndarray):
        return TensorType.NUMPY
    elif is_torch_available() and (torch.is_tensor(tensor) or isinstance(tensor, DeviceView)):
        return TensorType.TORCH
    elif is_tf_available() and tf.is_tensor(tensor):
        return TensorType.TENSORFLOW
    elif is_jax_available() and isinstance(tensor, (jax.numpy.ndarray, jaxlib.xla_extension.ArrayImpl)):
        return TensorType.JAX
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")


FRAMEWORK_TO_TENSOR_TYPE = {
    Framework.TORCH: TensorType.TORCH,
    Framework.TENSORFLOW: TensorType.TENSORFLOW,
    Framework.JAX: TensorType.JAX,
    Framework.ONNX: TensorType.NUMPY,
    Framework.NONE: TensorType.NUMPY,
    Framework.TENSORRT: TensorType.NUMPY,
}
