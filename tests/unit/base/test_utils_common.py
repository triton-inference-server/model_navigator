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
from model_navigator.utils.common import optimal_batch_size


def test_opt_batch_size_return_valid_result_when_various_max_bs_passed():
    pairs = [
        (1, 1),
        (2, 2),
        (3, 2),
        (5, 4),
        (16, 16),
        (31, 16),
        (32, 16),
        (35, 16),
        (63, 16),
        (512, 256),
        (4096, 1024),
        (16348, 2048),
        (65536, 8192),
    ]
    for max_bs, opt_bs in pairs:
        result = optimal_batch_size(max_batch_size=max_bs)
        assert result == opt_bs
