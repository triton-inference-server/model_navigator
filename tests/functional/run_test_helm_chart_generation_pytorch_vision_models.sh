#!/usr/bin/env bash
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

set -x

TEST_CONFIG="${1:-./tests/functional/pytorch_vision_models}"
CURRENT_DATE=$(date "+%Y%m%d_%H%M%S")

./tests/functional/test_helm_chart_generation.py -vvv \
    --work-dir ${CURRENT_DATE}_workspace_pytorch_vision_helm \
    --downloader-path tests/functional/pytorch_vision_models/pytorch_downloader.py \
    ${TEST_CONFIG} 2>&1|tee tee ${CURRENT_DATE}_workspace_pytorch_vision_helm.log
