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

from model_navigator.framework_api.package_utils import is_tf_available, is_torch_available
from model_navigator.framework_api.pipelines.pipeline import Pipeline  # noqa: F401
from model_navigator.framework_api.pipelines.pipeline_manager_base import PipelineManager  # noqa: F401

if is_torch_available():
    from model_navigator.framework_api.pipelines.pipeline_manager_pyt import TorchPipelineManager  # noqa: F401

if is_tf_available():
    from model_navigator.framework_api.pipelines.pipeline_manager_tf import TFPipelineManager  # noqa: F401
