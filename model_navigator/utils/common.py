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
import glob
import os
import pathlib
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy
from polygraphy.backend.trt.profile import Profile, ShapeTuple

from model_navigator.core.logger import LOGGER
from model_navigator.utils import module

torch = module.lazy_import("torch")

T = TypeVar("T")

PYTHON_PRIMITIVE_TYPES = (int, float, bool, bytes, type(None))


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
            elif isinstance(value_, pathlib.Path):
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
        if filter_fields:
            filtered_data = DataObject.filter_data(
                data=self.__dict__,
                filter_fields=filter_fields,
            )
        else:
            filtered_data = self.__dict__

        if parse:
            data = DataObject.parse_data(filtered_data)
        else:
            data = filtered_data

        return data

    @staticmethod
    def filter_data(data: Dict, filter_fields: List[str]):
        """Filter fields in dictionary.

        Args:
            data: Dictionary with data to filter
            filter_fields: Fields to filter

        Returns:
            Filtered dictionary
        """
        filtered_data = {key: value for key, value in data.items() if key not in filter_fields}
        return filtered_data

    @staticmethod
    def parse_data(data: Dict):
        """Parse values in provided data.

        Args:
            data: Dictionary with data to parse

        Returns:
            Parsed dictionary
        """
        parsed_data = {}
        for key, value in data.items():
            parsed_data[key] = DataObject.parse_value(value)

        return parsed_data

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


def _get_numpy_to_torch_dtype_dict():
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
    }


def numpy_to_torch_dtype(np_dtype: numpy.dtype) -> "torch.dtype":
    """Cast numpy dtype to torch dtype.

    Args:
        np_dtype (numpy.dtype): numpy dtype.

    Returns:
        torch.dtype: torch dtype.
    """
    np_dtype = numpy.dtype(np_dtype).type
    numpy_to_torch_dtype_dict = _get_numpy_to_torch_dtype_dict()
    return numpy_to_torch_dtype_dict[np_dtype]


def torch_to_numpy_dtype(torch_dtype: "torch.dtype") -> Type[numpy.dtype]:
    """Cast torch dtype to numpy dtype.

    Args:
        torch_dtype (torch.dtype): torch dtype.

    Returns:
        numpy.dtype: numpy dtype.
    """
    numpy_to_torch_dtype_dict = _get_numpy_to_torch_dtype_dict()
    torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}
    return torch_to_numpy_dtype_dict[torch_dtype]


def get_default_status_filename() -> str:
    """Get default status filename.

    Returns:
        str: Filename.
    """
    return "status.yaml"


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


def find_str_in_iterable(name, seq, index=None):
    """Search for string in sequence.

    Attempts to find matching strings in a sequence. Checks for exact matches, then
    case-insensitive substring matches, finally falling back to index based matching.

    Args:
        name: The key to search for.
        seq: The dictionary to search in.
        index: An index to fall back to if the string could not be found.

    Returns:
        str: The element found in the sequence, or None if it could not be found.
    """
    if name in seq:
        return name

    for elem in seq:
        if name.lower() in elem.lower() or elem.lower() in name.lower():
            return elem

    if index is not None and index < len(seq):
        return list(seq)[index]
    return None


def volume(obj):
    """Calculate the volume of an object.

    Returns: The volume of the object.
    """
    vol = 1
    for elem in obj:
        vol *= elem
    return vol


def default(value, default):
    """Returns a specified default value if the provided value is None.

    Args:
        value : The value.
        default : The default value to use if value is None.

    Returns:
        object: Either value, or the default.
    """
    return value if value is not None else default


def invoke_if_callable(func, *args, **kwargs):
    """Invoke if callable.

    Attempts to invoke a function with arguments. If `func` is not callable, then returns `func`
    The second return value of this function indicates whether the argument was a callable.
    """
    if callable(func):
        ret = func(*args, **kwargs)
        return ret, True
    return func, False


def warn_if_wrong_mode(file_like, mode: str):
    """Warn if file-like object has a different mode than requested."""

    # pytype: disable=attribute-error
    def binary(mode):
        return "b" in mode

    def readable(mode):
        return "r" in mode or "+" in mode

    def writable(mode):
        return "w" in mode or "a" in mode or "+" in mode

    fmode = file_like.mode
    if (
        binary(fmode) != binary(mode)
        or (readable(mode) and not readable(fmode))
        or (writable(mode) and not writable(fmode))
    ):
        LOGGER.warning(
            f"File-like object has a different mode than requested!\nNote: Requested mode was: {mode} but file-like object has mode: {file_like.mode}"
        )
    # pytype: enable=attribute-error


def is_file_like(obj):
    """Check if an object is file-like.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is file-like, False otherwise.
    """
    try:
        obj.read  # pytype: disable=attribute-error #noqa: B018
        obj.write  # pytype: disable=attribute-error #noqa: B018
    except AttributeError:
        return False
    else:
        return True


def load_file(
    src: Union[str, pathlib.Path], mode: str = "rb", description: Optional[str] = None
) -> Union[str, bytes, None]:
    """Reads from the specified source path or file-like object.

    Args:
        src: The path or file-like object to read from.


        mode: The mode to use when reading. Defaults to "rb".
        description: A description of what is being read.

    Returns:
        The contents read.

    Raises:
        Exception: If the file or file-like object could not be read.
    """
    if description is not None:
        LOGGER.info(f"Loading {description} from {src}")
    # pytype: disable=attribute-error
    if is_file_like(src):
        warn_if_wrong_mode(src, mode)
        # Reset cursor position after reading from the beginning of the file.
        prevpos = src.tell()
        if src.seekable():
            src.seek(0)
        contents = src.read()
        if src.seekable():
            src.seek(prevpos)
        return contents
    else:
        with open(src, mode) as f:
            return f.read()
    # pytype: enable=attribute-error


class BytesFromPath:
    """Functor that can load a file in binary mode ('rb')."""

    def __init__(self, path):
        """Loads a file in binary mode ('rb').

        Args:
            path (str): The file path.
        """
        self._path = path

    def __call__(self, *args, **kwargs):
        """Invokes the loader by forwarding arguments to ``call_impl``.

        Note: ``call_impl`` should *not* be called directly - use this function instead.
        """
        return self.call_impl(*args, **kwargs)

    def call_impl(self):
        """Implementation of ``__call__``.

        Returns:
            bytes: The contents of the file.
        """
        return load_file(self._path, description="bytes")


def is_contiguous(array):
    """Checks whether the provided NumPy array is contiguous in memory.

    Args:
        array (np.ndarray): The NumPy array.

    Returns:
        bool: Whether the array is contiguous in memory.
    """
    return array.flags["C_CONTIGUOUS"]


def make_contiguous(array):
    """Makes a NumPy array contiguous if it's not already.

    Args:
        array (np.ndarray): The NumPy array.

    Returns:
        np.ndarray: The contiguous NumPy array.
    """
    if not is_contiguous(array):
        return numpy.ascontiguousarray(array)
    return array


def resize_buffer(buffer: numpy.ndarray, shape: Sequence[int]) -> numpy.ndarray:
    """Resize a buffer to the specified shape.

    Resizes the provided buffer and makes it contiguous in memory,
    possibly reallocating the buffer.

    Args:
        buffer: The buffer to resize.
        shape: The desired shape of the buffer.

    Returns:
        The resized buffer, possibly reallocated.
    """
    if shape != buffer.shape:
        try:
            buffer.resize(shape, refcheck=False)
        except ValueError as err:
            LOGGER.warning(
                f"Could not resize host buffer to shape: {shape}. "
                f"Allocating a new buffer instead.\nNote: Error was: {err}"
            )
            buffer = numpy.empty(shape, dtype=numpy.dtype(buffer.dtype))
    return make_contiguous(buffer)


def find_in_dirs(name_glob: str, dirs: Sequence[str]) -> List[str]:
    """Finds a file, optionally including a glob expression, in the specified directories.

    Args:
        name_glob: The name of the file, optionally including a glob expression.
                Only the first match will be returned.
        dirs: The directories in which to search.

    Returns:
        The paths found, or an empty list if it could not be found.
    """
    for dir_name in dirs:
        paths = glob.glob(os.path.join(dir_name, name_glob))
        if paths:
            return paths
    return []
