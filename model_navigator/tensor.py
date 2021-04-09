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
import abc
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import attr
import numpy as np
import yaml


# pytype: disable=annotation-type-mismatch
# pytype: disable=wrong-keyword-args
# pytype: disable=name-error


def _is_dim_correct(self, attribute_name, dim):
    # int equal to 1 or positive number
    if not isinstance(dim, int) or (dim != -1 and dim <= 0):
        raise TypeError(f"Shape items should be integers equal to -1 or positive numbers. Got {dim}")


@attr.s(frozen=True)
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

    name: str = attr.ib(validator=[attr.validators.instance_of(str)])
    shape: Tuple = attr.ib(
        validator=[
            attr.validators.deep_iterable(
                member_validator=_is_dim_correct,
                iterable_validator=attr.validators.instance_of(tuple),
            )
        ],
        kw_only=True,
    )
    dtype: Optional[np.dtype] = attr.ib(
        validator=[attr.validators.optional(attr.validators.instance_of(np.dtype))], kw_only=True, default=None
    )

    def is_dynamic(self):
        """Check if tensor is dynamic - if any of dimension have -1 in shape. Except fist axis which is batch size."""
        return any([dim == -1 for dim in self.shape[1:]])

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
            shape=tuple([int(s) for s in tensor_metadata["shape"]]),
            dtype=np.dtype(client_utils.triton_to_np_dtype(tensor_metadata["datatype"])),
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
            shape=tuple([dim if isinstance(dim, int) else -1 for dim in metadata.shape]),
            dtype=metadata.dtype,
        )

    @classmethod
    def from_command_line(cls, shape_spec: str, delimiter: str = ":"):
        name, shape, *dtype = shape_spec.split(delimiter)
        shape = tuple(map(int, shape.split(",")))
        dtype = np.dtype(dtype[0]) if dtype else None
        return cls(name=name, shape=shape, dtype=dtype)

    def to_command_line(self, delimiter: str = ":"):
        name = self.name
        shape = ",".join(map(str, self.shape))

        parts = [name, shape]

        if self.dtype:
            parts.append(self.dtype.name)

        cli_value = f"{delimiter}".join(parts)
        return cli_value


@attr.s
class IOSpec:
    """Model inputs and outputs specification.

    Example usage:
    >>> io_spec = IOSpec.from_file("/tmp/model.json")
    >>> io_spec.write("/tmp/model.json")
    """

    inputs: Dict[str, TensorSpec] = attr.ib(
        validator=attr.validators.deep_mapping(
            key_validator=attr.validators.instance_of(str),
            value_validator=attr.validators.instance_of(TensorSpec),
            mapping_validator=attr.validators.instance_of(Dict),
        ),
        factory=list,
    )
    outputs: Dict[str, TensorSpec] = attr.ib(
        validator=attr.validators.deep_mapping(
            key_validator=attr.validators.instance_of(str),
            value_validator=attr.validators.instance_of(TensorSpec),
            mapping_validator=attr.validators.instance_of(Dict),
        ),
        factory=list,
    )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "IOSpec":
        file_path = Path(file_path)

        with file_path.open("r") as fh:
            io_spec_dict = yaml.load(fh, Loader=yaml.SafeLoader)

        def _wrap_entry(entry):
            return TensorSpec(entry["name"], shape=tuple(entry["shape"]), dtype=np.dtype(entry["dtype"]))

        inputs = {name: _wrap_entry(entry) for name, entry in io_spec_dict["inputs"].items()}
        outputs = {name: _wrap_entry(entry) for name, entry in io_spec_dict["outputs"].items()}
        return cls(inputs=inputs, outputs=outputs)

    def write(self, file_path: Union[str, Path]):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        def _extract_attribute(self_, name, value):
            if isinstance(value, np.dtype):
                value = value.name
            return value

        with file_path.open("w") as fh:
            io_specs = attr.asdict(self, value_serializer=_extract_attribute)
            yaml.dump(io_specs, fh, indent=4)


class TensorUtils:
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