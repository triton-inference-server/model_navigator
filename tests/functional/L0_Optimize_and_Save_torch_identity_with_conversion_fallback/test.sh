#!/usr/bin/env bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

set -ex

THIS_SCRIPT_PATH="$(realpath --relative-to="$(pwd)" "$0")"
TEST_MODULE="$(dirname "${THIS_SCRIPT_PATH}"|sed 's/\//./g').test"

export NAVIGATOR_CONSOLE_OUTPUT=LOGS
export NAVIGATOR_LOG_LEVEL=DEBUG

python -m"${TEST_MODULE}" \
    --status $(pwd)/status.yaml

