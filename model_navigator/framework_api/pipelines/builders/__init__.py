# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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


from model_navigator.framework_api.package_utils import is_jax_available, is_tf_available, is_torch_available
from model_navigator.framework_api.pipelines.builders.config_generation import config_generation_builder  # noqa: F401
from model_navigator.framework_api.pipelines.builders.correctness import correctness_builder  # noqa: F401
from model_navigator.framework_api.pipelines.builders.preprocessing import preprocessing_builder  # noqa: F401
from model_navigator.framework_api.pipelines.builders.profiling import profiling_builder  # noqa: F401

if is_torch_available():
    from model_navigator.framework_api.pipelines.builders.torch import (  # noqa: F401
        torch_conversion_builder,
        torch_export_builder,
    )

if is_tf_available():
    from model_navigator.framework_api.pipelines.builders.tensorflow import (  # noqa: F401;
        tensorflow_conversion_builder,
        tensorflow_export_builder,
    )

if is_tf_available() and is_jax_available():
    from model_navigator.framework_api.pipelines.builders.jax import jax_export_builder  # noqa: F401

from model_navigator.framework_api.pipelines.builders.onnx import (  # noqa: F401
    onnx_conversion_builder,
    onnx_export_builder,
)
