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
# pytype: skip-file

import pathlib
from enum import Enum
from typing import Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy
from polygraphy.common import TensorMetadata as PolygraphyTensorMetadata

from model_navigator.tensor import TensorSpec

Sample = Dict[str, numpy.ndarray]


class Format(Enum):
    TORCHSCRIPT_SCRIPT = "torchscript-script"
    TORCHSCRIPT_TRACE = "torchscript-trace"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TORCH_TRT_SCRIPT = "torch-trt-script"
    TORCH_TRT_TRACE = "torch-trt-trace"
    ONNX = "onnx"
    TENSORRT = "trt"


class DataObject:
    def to_dict(self, filter_fields: Optional[List[str]] = None, parse: bool = False):
        data = {}

        if filter_fields:
            filtered_data = {key: value for key, value in self.__dict__.items() if key not in filter_fields}
        else:
            filtered_data = self.__dict__

        if parse:
            for key, value in filtered_data.items():
                if value is None:
                    continue
                data[key] = self._parse_value(value)
        else:
            data = filtered_data

        return data

    def _parse_value(self, value):
        if isinstance(value, DataObject):
            value = value.to_dict(parse=True)
        elif hasattr(value, "to_json"):
            value = value.to_json()
        elif isinstance(value, Mapping):
            value = self._from_dict(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            value = self._from_list(value)
        elif isinstance(value, Enum):
            value = value.value
        elif isinstance(value, pathlib.Path):
            value = str(value)
        return value

    def _from_dict(self, values):
        data = {}
        for key, value in values.items():
            data[key] = self._parse_value(value)

        return data

    def _from_list(self, values):
        items = []
        for value in values:
            item = self._parse_value(value)
            items.append(item)

        return items


class TensorMetadata(Dict[str, TensorSpec]):
    def add(self, name, shape, dtype):
        self[name] = TensorSpec(name, tuple(shape), numpy.dtype(dtype))

    @classmethod
    def from_json(cls, data: List):
        tensor_metadata = cls()
        for value in data:
            tensor_metadata.add(value["name"], value["shape"], value["dtype"])
        return tensor_metadata

    def to_json(self):
        data = []
        for spec in self.values():
            data.append(self._parse_tensorspec(spec))
        return data

    @staticmethod
    def _parse_tensorspec(spec: TensorSpec):
        return {"name": spec.name, "shape": spec.shape, "dtype": str(spec.dtype)}

    @classmethod
    def from_polygraphy_tensor_metadata(cls, polygraphy_tensor_metadata: PolygraphyTensorMetadata):
        tensor_metadata = cls()
        for name, data in polygraphy_tensor_metadata.items():
            shape = [d if isinstance(d, int) else -1 for d in data.shape]
            tensor_metadata.add(name, shape, data.dtype)
        return tensor_metadata


@runtime_checkable
class SizedIterable(Protocol):
    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...


SizedDataLoader = Union[SizedIterable, Sequence]
