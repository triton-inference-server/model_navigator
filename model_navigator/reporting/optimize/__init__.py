# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Events module."""

from model_navigator.reporting.optimize.detailed_report import DetailedReport
from model_navigator.reporting.optimize.simple_report import SimpleReport
from model_navigator.utils.environment import get_console_output

_reporter = None


def initialize_optimize_reporting():
    """Initialize reporting."""
    if get_console_output() in ["DETAILED", "LOGS"]:
        _reporter = DetailedReport()
    else:
        _reporter = SimpleReport()
