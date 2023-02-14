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
"""TensorRT runners."""
from typing import List

from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes
from polygraphy.backend.trt import TrtRunner as PolygraphyTrtRunner

from model_navigator.api.config import Format
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils import tensorrt


class TensorRTRunner(NavigatorRunner):
    """Runs inference using PyTorch."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialization implementation."""
        super().__init__(*args, **kwargs)
        self._runner = PolygraphyTrtRunner(
            engine=EngineFromBytes(BytesFromPath(self.model.as_posix())),
            name=self.name(),
            optimization_profile=None,
        )

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TENSORRT

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TensorRT"

    def activate_impl(self):
        """Activate implementation."""
        self._runner.activate_impl()

    def infer_impl(self, feed_dict):
        """Inference implementation."""
        input_metadata = self._runner.get_input_metadata()
        feed_dict = {name: tensorrt.cast_tensor(tensor) for name, tensor in feed_dict.items() if name in input_metadata}
        return self._runner.infer_impl(feed_dict)

    def deactivate_impl(self):
        """Deactivate implementation."""
        self._runner.deactivate_impl()


def register_tensorrt_runners():
    """Register TensorRT runners."""
    register_runner(TensorRTRunner)
