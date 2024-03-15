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
"""Enums utils."""

from typing import Callable, Iterable, Tuple, TypeVar, Union

T = TypeVar("T")
V = TypeVar("V")


def parse(value: Union[V, Iterable[V]], parse_func: Callable[[V], T]) -> Tuple[T, ...]:
    """Parse a value or an iterable of values to a tuple of parsed values.

    Args:
        value (Union[V, Iterable[V]]): Value or iterable of values.
        parse_func (Callable[[V], T]): Parse function that parse a value.

    Returns:
        Tuple[T, ...]: Tuple of parsed values.
    """
    if value is not None:
        value = tuple(value) if isinstance(value, (tuple, list)) else (value,)
        value = tuple(parse_func(v) for v in value)
        return value
    return ()
