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
import numpy as np
import pytest

from model_navigator.commands.export.torch import _validate_if_dynamic_axes_aligns_with_dataloader_shapes
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.utils.tensor import TensorMetadata


def test_validate_if_dynamic_axes_aligns_with_dataloader_shapes_raises_error_when_unknown_axes():
    dynamic_axes = {"unknown_input": [0]}
    input_metadata = TensorMetadata().add("input", [1, 2], np.float32)
    output_metadata = TensorMetadata().add("output", [1, 2], np.float32)

    with pytest.raises(ModelNavigatorConfigurationError):
        _validate_if_dynamic_axes_aligns_with_dataloader_shapes(dynamic_axes, input_metadata, output_metadata)


def test_validate_if_dynamic_axes_aligns_with_dataloader_shapes_raises_error_when_missing_dataloader_dynamic_ax():
    dynamic_axes = {"input": [0]}
    input_metadata = TensorMetadata().add("input", [-1, -1], np.float32)
    output_metadata = TensorMetadata().add("output", [1, 2], np.float32)

    with pytest.raises(ModelNavigatorConfigurationError):
        _validate_if_dynamic_axes_aligns_with_dataloader_shapes(dynamic_axes, input_metadata, output_metadata)


def test_validate_if_dynamic_axes_aligns_with_dataloader_shapes_raises_no_errors_when_dynamic_axes_aligns():
    dynamic_axes = {"input": [0, 1]}
    input_metadata = TensorMetadata().add("input", [-1, -1], np.float32)
    output_metadata = TensorMetadata().add("output", [1, 2], np.float32)

    _validate_if_dynamic_axes_aligns_with_dataloader_shapes(dynamic_axes, input_metadata, output_metadata)
