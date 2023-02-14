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
"""Pipelines utils."""
from typing import Union

from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import (
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TorchTensorRTConfig,
)


def do_run_max_batch_size_search(
    config: CommonConfig,
    model_cfg: Union[TensorRTConfig, TensorFlowTensorRTConfig, TorchTensorRTConfig],
) -> bool:
    """Should max batch size search be run for the model.

    Args:
        config: Common optimize configuration.
        model_cfg: Model configuration.

    Returns:
        bool: True if run max batch size.
    """
    return bool(model_cfg.trt_profile) is False and config.batch_dim is not None
