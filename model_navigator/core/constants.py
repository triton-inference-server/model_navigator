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
"""Constants definition."""

from model_navigator.__version__ import __version__  # noqa: F401

# Version
NAVIGATOR_VERSION = __version__
NAVIGATOR_PACKAGE_VERSION = "0.3.1"

# Workspace related
DEFAULT_WORKSPACE = "navigator_workspace"

# Profiling related
DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD = 0.05
DEFAULT_PROFILING_LATENCY_CUTOFF_THRESHOLD = 0.1

# Dataloader related
DEFAULT_SAMPLE_COUNT = 100

# TensorRT conversion related
DEFAULT_MAX_WORKSPACE_SIZE = 8589934592
DEFAULT_MIN_SEGMENT_SIZE = 3
DEFAULT_TENSORRT_MAX_DIMENSION_SIZE = 2**31 - 1
OPT_MAX_SHAPE_RATIO = 4 / 5

# ONNX export/conversion related
DEFAULT_ONNX_OPSET = 17

# Find Max Batch Size
DEFAULT_MAX_BATCH_SIZE_THRESHOLD = 512
DEFAULT_MAX_BATCH_SIZE_HALVING = 2

# Logging
NAVIGATOR_LOGGER_NAME = "Navigator"
NAVIGATOR_LOG_NAME = "navigator.log"

# Timer
DEFAULT_COMPARISON_REPORT_FILE = "report.yaml"
