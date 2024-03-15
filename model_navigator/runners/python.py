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

from typing import Dict, List

from model_navigator.api.config import Format
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner


class PythonRunner(NavigatorRunner):
    """Runs inference for Python models."""

    def infer_impl(self, feed_dict: Dict, *args, **kwargs):
        """Runner inference handler implementation."""
        inputs = self.input_metadata.unflatten_sample(feed_dict, wrap_input=True)
        if isinstance(inputs[-1], dict):
            args, kwargs = inputs[:-1], inputs[-1]
        else:
            args, kwargs = inputs, {}

        outputs = self.model(*args, **kwargs)

        if self.output_metadata is None:
            return outputs

        out_dict = self.output_metadata.flatten_sample(outputs)
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
