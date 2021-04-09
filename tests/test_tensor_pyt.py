# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# pytype: disable=import-error
import torch
from model_navigator.tensor import NPTensorUtils, TensorUtils

# pytype: enable=import-error


def test_pyt_eq():
    a = torch.ones((8, 64), dtype=torch.float32).to("cpu:0")
    utils = TensorUtils.for_data(a)
    assert utils.eq(a, a)

    b = a.clone().detach()
    assert utils.eq(a, b)

    # on different devices
    c = a.clone().detach().to("cuda:0")
    assert not utils.eq(a, c)

    # simple change
    d = a.clone().detach()
    d[0, 0] = 0
    assert not utils.eq(a, d)

    # different dtypes should return False
    e = a.clone().detach().double()
    assert not utils.eq(a, e)

    # different shapes also should return False
    f = torch.ones((9, 64), dtype=torch.float32).to("cpu:0")
    assert not utils.eq(a, f)


def test_pyt_to_numpy():
    a = torch.ones((8, 64), dtype=torch.float32).to("cuda:0")
    utils = TensorUtils.for_data(a)
    b = utils.to_numpy(a)
    assert NPTensorUtils.eq(b, np.ones((8, 64), dtype=np.float32))

    a[0, 0] = np.nan
    c = utils.to_numpy(a)
    c_expected = np.ones(a.shape, dtype=np.float32)
    c_expected[0, 0] = np.nan
    assert NPTensorUtils.eq(c, c_expected)
