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
"""Runner definition for Python based models."""
from collections import OrderedDict
from typing import Dict, List, Mapping

import numpy

from model_navigator.api.config import Format
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils.dataloader import get_default_output_names


class PythonRunner(NavigatorRunner):
    """Runs inference for Python models."""

    def infer_impl(self, feed_dict: Dict):
        """Runner inference handler implementation."""
        outputs = self.model(**feed_dict).values()

        if self.output_metadata:
            output_names = self.output_metadata.keys()
        else:
            output_names = outputs.keys() if isinstance(outputs, Mapping) else get_default_output_names(len(outputs))

        if isinstance(outputs, numpy.ndarray):
            outputs = (outputs,)
        if isinstance(outputs, Mapping):
            outputs = outputs.values()

        out_dict = OrderedDict()
        for name, output in zip(output_names, outputs):
            out_dict[name] = output
        return out_dict

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.PYTHON

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CPU, DeviceKind.CUDA]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "PythonRunner"


def register_python_runners():
    """Register Python runner in global registry."""
    register_runner(PythonRunner)
