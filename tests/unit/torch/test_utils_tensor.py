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

import torch  # pytype: disable=import-error

from model_navigator.configuration import TensorType
from model_navigator.core.tensor import get_tensor_type


def test_get_tensor_type_torch():
    a = torch.ones((8, 64), dtype=torch.float32)
    assert get_tensor_type(a) == TensorType.TORCH
