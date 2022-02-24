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

from typing import Dict, Iterator, Protocol, Sequence, Union

import numpy

from model_navigator.tensor import TensorSpec

Sample = Dict[str, numpy.ndarray]


class TensorMetadata(Dict[str, TensorSpec]):
    def add(self, name, shape, dtype):
        self[name] = TensorSpec(name, shape, dtype)


class SizedIterable(Protocol):
    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...


SizedDataLoader = Union[SizedIterable, Sequence]
