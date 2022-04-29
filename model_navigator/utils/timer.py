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
import time
from typing import Optional

from model_navigator.exceptions import ModelNavigatorException


class Timer:
    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self):
        """Start measurement"""
        if self._start:
            raise ModelNavigatorException("Timer has been already started. User reset to perform new measurement.")

        self._start = time.monotonic()

    def stop(self):
        """End measurement"""
        if not self._start:
            raise ModelNavigatorException("Timer has been not started.")

        self._end = time.monotonic()

    def reset(self):
        """Reset measurement points"""
        self._start = None
        self._end = None

    def duration(self):
        """Calculate and return duration in seconds"""
        if self._start is None or self._end is None:
            raise ModelNavigatorException("Timer has been not started or stopped.")

        return self._end - self._start
