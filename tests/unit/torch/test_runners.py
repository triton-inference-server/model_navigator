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
"""Test for Torch API"""
from distutils.version import LooseVersion

import numpy as np
import pytest

from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.runners.torch import TorchTensorRTRunner
from model_navigator.utils import module

torch = module.lazy_import("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_tensorrt_to_torch_dtype_cast_int64():
    torch_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    casted_tensor = TorchTensorRTRunner(model=None, input_metadata=None, output_metadata=None)._to_torch_tensor(
        torch_tensor, np.int64
    )
    if tensorrt_utils.get_version() > LooseVersion("8.6"):
        assert casted_tensor.dtype == torch.int64
    else:
        assert casted_tensor.dtype == torch.int32

    numpy_tensor = np.array([1, 2, 3], dtype=np.int64)
    casted_tensor = TorchTensorRTRunner(model=None, input_metadata=None, output_metadata=None)._to_torch_tensor(
        numpy_tensor, np.int64
    )
    if tensorrt_utils.get_version() > LooseVersion("8.6"):
        assert casted_tensor.dtype == torch.int64
    else:
        assert casted_tensor.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_tensorrt_to_torch_dtype_cast_float64():
    torch_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
    casted_tensor = TorchTensorRTRunner(model=None, input_metadata=None, output_metadata=None)._to_torch_tensor(
        torch_tensor, np.float64
    )
    assert casted_tensor.dtype == torch.float32

    numpy_tensor = np.array([1, 2, 3], dtype=np.float64)
    casted_tensor = TorchTensorRTRunner(model=None, input_metadata=None, output_metadata=None)._to_torch_tensor(
        numpy_tensor, np.float64
    )
    assert casted_tensor.dtype == torch.float32
