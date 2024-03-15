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
"""Inplace Optimize utility functions."""

import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional

from model_navigator.commands.infer_metadata import _get_trt_profile_from_axes_shapes
from model_navigator.core.tensor import PyTreeMetadata
from model_navigator.utils.module import lazy_import

torch = lazy_import("torch")


def get_object_name(obj: Any) -> str:
    """Get the name of an object from its module and class."""
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"


class TorchDataloader:
    """Dataloader for torch models.

    Yields samples read from disk.
    """

    def __init__(self, samples_paths: List[pathlib.Path]) -> None:
        """Initialize TorchDataloader.

        Args:
            samples_paths: list of paths to samples.
        """
        self._samples_paths = samples_paths

    def __iter__(self):
        """Iterate over samples."""
        for sample_path in self._samples_paths:
            yield torch.load(sample_path)

    def __len__(self):
        """Get number of samples."""
        return len(self._samples_paths)


def _extract_axes_shapes(
    shapes: List[Dict[str, List[int]]],
    pytree_metadata: PyTreeMetadata,
) -> Dict[str, Dict[int, List[int]]]:
    axes_shapes = {name: defaultdict(list) for name in pytree_metadata.get_names()}
    for sample_shapes in shapes:
        for name, tensor_shape in sample_shapes.items():
            for k, dim in enumerate(tensor_shape):
                axes_shapes[name][k].append(dim)

    return axes_shapes


def get_trt_profile_from_shapes(
    shapes: List[Dict[str, List[int]]],
    pytree_metadata: PyTreeMetadata,
    batch_dim: Optional[int],
    max_batch_size: Optional[int],
):
    """Get trt profile from shapes."""
    axes_shapes = _extract_axes_shapes(shapes, pytree_metadata)
    return _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim, max_batch_size)


def get_dynamic_axes_from_shapes(
    shapes: List[Dict[str, List[int]]], pytree_metadata: PyTreeMetadata, batch_dim: Optional[int]
):
    """Get dynamic axes from shapes."""
    axes_shapes = _extract_axes_shapes(shapes, pytree_metadata)
    dynamic_axes = defaultdict(list)
    for name, axes_shapes_ in axes_shapes.items():
        for axis, shapes in axes_shapes_.items():
            if axis == batch_dim or min(shapes) != max(shapes):
                dynamic_axes[name].append(axis)
    return dict(dynamic_axes)
