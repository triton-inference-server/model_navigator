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

from typing import Callable, Dict, List

from model_navigator.api.config import Format
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.frameworks import is_jax_available, is_tf_available, is_torch_available
from model_navigator.pipelines.builders.correctness import correctness_builder  # noqa: F401
from model_navigator.pipelines.builders.performance import performance_builder  # noqa: F401
from model_navigator.pipelines.builders.preprocessing import preprocessing_builder  # noqa: F401
from model_navigator.pipelines.builders.verify import verify_builder  # noqa: F401
from model_navigator.pipelines.pipeline import Pipeline  # noqa: F401

if is_torch_available():
    from .torch import torch_conversion_builder, torch_export_builder  # noqa: F401
    from .torch_tensorrt import torch_tensorrt_conversion_builder  # noqa: F401

if is_tf_available():
    from .tensorflow import tensorflow_conversion_builder, tensorflow_export_builder  # noqa: F401;
    from .tensorflow_tensorrt import tensorflow_tensorrt_conversion_builder  # noqa: F401

if is_tf_available() and is_jax_available():
    from .jax import jax_export_builder  # noqa: F401

from .tensorrt import tensorrt_conversion_builder  # noqa: F401

PipelineBuilder = Callable[[CommonConfig, Dict[Format, List[ModelConfig]]], Pipeline]
