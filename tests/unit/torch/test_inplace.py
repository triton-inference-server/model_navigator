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

import pytest
import torch  # pytype: disable=import-error
from torch.nn import Linear  # pytype: disable=import-error

from model_navigator.inplace.config import OptimizeConfig, inplace_config
from model_navigator.inplace.model import OptimizedModule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")
def test_optimized_model_releases_torch_cuda_memory(tmp_path):
    # given
    initial_memory_allocated = torch.cuda.memory_allocated()
    initial_memory_reserved = torch.cuda.memory_reserved()
    module = Linear(1000, 1000, device="cuda")
    module_name = "module"
    inplace_config.cache_dir = tmp_path
    (tmp_path / module_name).mkdir(parents=True, exist_ok=True)
    # when
    wrapped = OptimizedModule(
        module=module,
        name=module_name,
        input_mapping=lambda x: x,
        output_mapping=lambda x: x,
        optimize_config=OptimizeConfig(),
    )
    wrapped._offload_module()

    # then
    assert torch.cuda.memory_allocated() == initial_memory_allocated  # did not grow
    # ideally reserved should be 0, but on some devices it does not drop to zero, use less strict checking i.e. <=
    assert torch.cuda.memory_reserved() <= initial_memory_reserved  # did not grow
