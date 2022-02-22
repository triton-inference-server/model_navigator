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

from typing import List

import numpy
from polygraphy.backend.base import BaseRunner

from model_navigator.framework_api.commands.core import Command, Performance


class PerformanceBase(Command):
    def _get_runner(self, **kwargs) -> BaseRunner:
        raise NotImplementedError

    def __call__(
        self,
        samples: List,
        **kwargs,
    ) -> Performance:

        runner = self._get_runner(samples=samples, **kwargs)

        time_measurements = []
        with runner:
            for sample in samples:
                runner.infer(sample)
                time_measurements.append(runner.last_inference_time())

        latency_sec = float(numpy.median(time_measurements))
        return Performance(latency_sec * 1000, 1 / latency_sec)
