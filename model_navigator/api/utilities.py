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
# noqa: D104
"""Public utilities for the Model Navigator API."""

from typing import Callable

from model_navigator.api.config import SizedDataLoader


class UnpackedDataloader:
    """A wrapper around a SizedDataLoader that applies a function to each sample.

    Args:
        dataloader: A SizedDataLoader.
        unpack_fn: A function that takes a sample and returns a new sample.

    Returns:
        An iterator over the samples in the dataloader with the unpack_fn applied.

    Example:
        >>> dataloader = [1, 2, 3]
        >>> unpacked_dataloader = UnpackedDataloader(dataloader, lambda x: x + 1)
        >>> # unpacked_dataloader is now [2, 3, 4]
    """

    def __init__(self, dataloader: SizedDataLoader, unpack_fn: Callable):
        """Initialize the UnpackedDataloader."""
        self._dataloader = dataloader
        self._unpack_fn = unpack_fn

    def __len__(self):
        """Return the number of samples in the dataloader."""
        return len(self._dataloader)

    def __iter__(self):
        """Return an iterator over the samples in the dataloader with the unpack_fn applied."""
        for sample in self._dataloader:
            yield self._unpack_fn(sample)
