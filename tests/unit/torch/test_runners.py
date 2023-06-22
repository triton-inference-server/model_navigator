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
import numpy as np
import pytest

from model_navigator.runners.torch import TorchTensorRTRunner
from model_navigator.utils import module

torch = module.lazy_import("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_tensorrt_to_torch_dtype_cast_int64_to_int32():
    torch_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    casted_tensor = TorchTensorRTRunner._to_torch_tensor(torch_tensor, np.int64)
    assert casted_tensor.dtype == torch.int32

    numpy_tensor = np.array([1, 2, 3], dtype=np.int64)
    casted_tensor = TorchTensorRTRunner._to_torch_tensor(numpy_tensor, np.int64)
    assert casted_tensor.dtype == torch.int32
