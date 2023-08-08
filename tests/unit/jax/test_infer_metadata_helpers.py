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

import numpy

from model_navigator.api.config import TensorType
from model_navigator.commands.infer_metadata import _extract_axes_shapes
from model_navigator.core.tensor import PyTreeMetadata
from model_navigator.frameworks import Framework


def test_extract_axes_shapes_return_correct_shapes_when_valid_dataloader_used():
    num_samples = 5
    shape = (1, 224, 224, 3)
    input_name = "input_0"
    expected_axes_shapes = {
        input_name: {0: [1, 1, 1, 1, 1], 1: [224, 224, 224, 224, 224], 2: [224, 224, 224, 224, 224], 3: [3, 3, 3, 3, 3]}
    }

    axes_shapes = _extract_axes_shapes(
        dataloader=[{input_name: numpy.full(shape=shape, fill_value=1)} for _ in range(num_samples)],
        pytree_metadata=PyTreeMetadata(metadata={"input_0": "input_0"}, tensor_type=TensorType.NUMPY),
        input_names=[input_name],
        input_ndims=[len(shape)],
        num_samples=num_samples,
        framework=Framework.JAX,
        check_len=True,
    )

    assert axes_shapes == expected_axes_shapes
