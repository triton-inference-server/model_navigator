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
"""Common utils."""

import dataclasses
import pathlib
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import dacite
import numpy
from polygraphy.backend.trt.profile import Profile, ShapeTuple

if TYPE_CHECKING:
    import torch  # pytype: disable=import-error

T = TypeVar("T")


def dataclass2dict(config: Any) -> Dict:
    """Parse a dataclass to a dictionary.

    Args:
        config: Dataclass to be parsed.

    Returns:
        Dict: Parsed dictionary.
    """
    assert dataclasses.is_dataclass(config), "`config` must be a dataclass."

    def _dict_factory_with_enum_values_extraction(fields_: Iterable[Tuple[Any, Any]]) -> Dict:
        result = []
        for key_, value_ in fields_:
            if isinstance(key_, Enum):
                key_ = key_.value

            if isinstance(value_, Enum):
                value_ = value_.value
            elif isinstance(value_, Path):
                value_ = value_.as_posix()
            elif isinstance(value_, numpy.dtype):
                value_ = str(value_)
            elif isinstance(value_, (tuple, list)):
                value_ = [v.value if isinstance(v, Enum) else v for v in value_]
            elif isinstance(value_, dict):
                value_ = _dict_factory_with_enum_values_extraction(value_.items())
            result.append((key_, value_))
        return dict(result)

    init_fields_names = [field.name for field in dataclasses.fields(config) if field.init]

    new_config_dict = dataclasses.asdict(config, dict_factory=_dict_factory_with_enum_values_extraction)
    config_dict_with_only_init_items = {k: v for k, v in new_config_dict.items() if k in init_fields_names}
    return config_dict_with_only_init_items


def dict2dataclass(cls: Type[T], data: Dict) -> T:
    """Parse a dictionary to a dataclass.

    Args:
        cls (Type[T]): Type of the dataclass.
        data (Dict): Dictionary with the dataclass data.

    Returns:
        T: Dataclass.
    """
    dataclass = dacite.from_dict(cls, data, config=dacite.Config(cast=[Enum, Path, tuple, numpy.dtype]))
    assert isinstance(dataclass, cls)
    assert dataclasses.is_dataclass(dataclass)
    return dataclass


class DataObject:
    """Class for storing data about configurations, statuses, etc."""

    def to_dict(self, filter_fields: Optional[List[str]] = None, parse: bool = False) -> Dict:
        """Serialize to a dictionary.

        Args:
            filter_fields (Optional[List[str]], optional): List of fields to filter out.
                Defaults to None.
            parse (bool, optional): If True recursively parse field values to jsonable representation.
                Defaults to False.

        Returns:
            Dict: Data serialized to a dictionary.
        """
        data = {}

        if filter_fields:
            filtered_data = {key: value for key, value in self.__dict__.items() if key not in filter_fields}
        else:
            filtered_data = self.__dict__

        if parse:
            for key, value in filtered_data.items():
                data[key] = self.parse_value(value)
        else:
            data = filtered_data

        return data

    @staticmethod
    def parse_value(value: Any) -> Union[str, Dict, List]:
        """Parse value to jsonable format.

        Args:
            value (Any): Value to be parsed.

        Returns:
            Union[str, Dict, List]: Jsonable value.
        """
        if isinstance(value, DataObject):
            value = value.to_dict(parse=True)
        elif hasattr(value, "to_json"):
            value = value.to_json()
        elif isinstance(value, (Mapping, Profile)):
            value = DataObject._from_dict(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            value = DataObject._from_list(value)
        elif isinstance(value, Enum):
            value = value.value
        elif isinstance(value, pathlib.Path):
            value = str(value)
        elif isinstance(value, ShapeTuple):
            value = vars(value)
        return value

    @staticmethod
    def _from_dict(values: Dict) -> Dict:
        data = {}
        for key, value in values.items():
            data[key] = DataObject.parse_value(value)

        return data

    @staticmethod
    def _from_list(values: List) -> List:
        items = []
        for value in values:
            item = DataObject.parse_value(value)
            items.append(item)

        return items


def parse_enum(value: Union[Union[str, T], Sequence[Union[str, T]]], enum_type: Type[T]) -> Tuple[T, ...]:
    """Parse string or a sequence of strings into a tuple of enums.

    Args:
        value: values to be parsed as enums.
        enum_type (Type[T]): Type of the enum

    Returns:
        Tuple[T, ...]: Tuple of enums.
    """
    if value is not None:
        value = tuple(value) if isinstance(value, (tuple, list)) else (value,)
        value = tuple(enum_type(v) for v in value)
        return value
    return ()


def numpy_to_torch_dtype(np_dtype: numpy.dtype) -> "torch.dtype":
    """Cast numpy dtype to torch dtype.

    Args:
        np_dtype (numpy.dtype): numpy dtype.

    Returns:
        torch.dtype: torch dtype.
    """
    np_dtype = numpy.dtype(np_dtype).type
    import torch  # pytype: disable=import-error

    return {
        numpy.bool_: torch.bool,
        numpy.uint8: torch.uint8,
        numpy.int8: torch.int8,
        numpy.int16: torch.int16,
        numpy.int32: torch.int32,
        numpy.int64: torch.int64,
        numpy.float16: torch.float16,
        numpy.float32: torch.float32,
        numpy.float64: torch.float64,
        numpy.complex64: torch.complex64,
        numpy.complex128: torch.complex128,
    }[np_dtype]


def get_default_status_filename() -> str:
    """Get default status filename.

    Returns:
        str: Filename.
    """
    return "status.yaml"


def get_default_workspace() -> Path:
    """Get default workspace path.

    Returns:
        Path: Worskspace path.
    """
    return Path.cwd() / "navigator_workspace"


def pad_string(s: str) -> str:
    """Pad string with `=` signs.

    Args:
        s (str): String.

    Returns:
        str: Padded string.
    """
    s = f"{30 * '='} {s} "
    s = s.ljust(100, "=")
    return s


def parse_kwargs_to_cmd(kwargs: Dict[str, Any]) -> List[str]:
    """Parse kwargs so that they can be passed as a commandline arguments.

    Args:
        kwargs (Dict[str, Any]): keyword arguments to be parsed to commandline.

    Returns:
        List[str]: List of commandline arguments.
    """
    args = []
    for k, v in kwargs.items():
        s = str(v).replace("'", '"')
        args.extend([f"--{k}", f"{s!r}"])
    return args
