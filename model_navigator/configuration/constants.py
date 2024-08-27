# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Constants definition for configuration."""

# Workspace related
DEFAULT_WORKSPACE = "navigator_workspace"

# Profiling related
DEFAULT_WINDOW_SIZE: int = 25
DEFAULT_MIN_TRIALS: int = 3
DEFAULT_MAX_TRIALS: int = 10
DEFAULT_STABILIZATION_WINDOWS: int = 3
DEFAULT_STABILITY_PERCENTAGE: float = 10.0
DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD = 0.05
DEFAULT_LATENCY_CUTOFF_THRESHOLD = 0.1
DEFAULT_THROUGHPUT_BACKOFF_LIMIT = 2

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
NAVIGATOR_LOG_FILENAME = "navigator.log"

# Logging env variables
NAVIGATOR_LOG_LEVEL_ENV = "NAVIGATOR_LOG_LEVEL"
NAVIGATOR_LOG_FORMAT_ENV = "NAVIGATOR_LOG_FORMAT"
NAVIGATOR_CONSOLE_OUTPUT_ENV = "NAVIGATOR_CONSOLE_OUTPUT"
NAVIGATOR_THIRD_PARTY_LOG_LEVEL_ENV = "NAVIGATOR_THIRD_PARTY_LOG_LEVEL"
OUTPUT_LOGS_FLAG = "LOGS"
OUTPUT_SIMPLE_REPORT = "SIMPLE"

# Timer
DEFAULT_COMPARISON_REPORT_FILE = "report.yaml"

# Subcommands isolation
NAVIGATOR_USE_MULTIPROCESSING = "NAVIGATOR_USE_MULTIPROCESSING"
