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

from model_navigator.configuration import Format, TensorRTConfig, TorchTensorRTConfig
from model_navigator.frameworks.torch.utils import update_allowed_batching_parameters


def test_update_allowed_batching_parameters_return_original_custom_configs_when_empty_none_provide():
    target_formats, custom_configs = update_allowed_batching_parameters(
        target_formats=(Format.TORCH,),
        custom_configs=None,
    )
    assert target_formats == (Format.TORCH,)
    assert custom_configs is None


def test_update_allowed_batching_parameters_remove_target_format_when_torch_trt_provided():
    target_formats, custom_configs = update_allowed_batching_parameters(
        target_formats=(
            Format.TORCH,
            Format.TORCH_TRT,
        ),
        custom_configs=None,
    )

    assert target_formats == (Format.TORCH,)
    assert custom_configs is None


def test_update_allowed_batching_parameters_remove_custom_config_when_torch_trt_provided():
    trt_config = TensorRTConfig()
    torch_trt_config = TorchTensorRTConfig()
    target_formats, custom_configs = update_allowed_batching_parameters(
        target_formats=(Format.TORCH,),
        custom_configs=[trt_config, torch_trt_config],
    )

    assert target_formats == (Format.TORCH,)
    assert custom_configs == [trt_config]
