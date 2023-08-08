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
from importlib.util import find_spec

import pytest

from model_navigator.commands.performance.nvml_handler import NvmlHandler


def _gpu_count() -> int:
    with NvmlHandler() as nvm_handler:
        return nvm_handler.gpu_count


def test_commands_performance_nvml_handler_not_initialized() -> None:
    nvml_handler = NvmlHandler()
    assert nvml_handler.gpu_clock is None
    assert nvml_handler.gpu_count == 0


@pytest.mark.skipif(
    _gpu_count() == 0 or not find_spec("torch"), reason="GPU is not available or PyTorch is not installed."
)
def test_commands_performance_nvml_handler_gpu_torch() -> None:
    import torch  # type: ignore

    with NvmlHandler() as nvm_handler:
        x = torch.rand((1000, 1000)).cuda()
        torch.matmul(x, x)
        torch.cuda.synchronize()
        assert nvm_handler.gpu_clock >= 0
        assert nvm_handler.gpu_count >= 0
