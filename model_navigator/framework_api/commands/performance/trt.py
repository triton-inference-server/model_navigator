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


from pathlib import Path
from typing import Tuple

from polygraphy.backend.base import BaseRunner
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.convert.onnx import ConvertONNX2TRT
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.performance.base import PerformanceBase
from model_navigator.framework_api.common import Format
from model_navigator.framework_api.runners.trt import TrtRunner
from model_navigator.framework_api.utils import get_package_path


class PerformanceTRT(PerformanceBase):
    def __init__(self, target_precision: TensorRTPrecision, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Performance TensorRT",
            command_type=CommandType.PERFORMANCE,
            target_format=Format.TENSORRT,
            requires=requires,
        )
        self.target_precision = target_precision

    def _get_runner(
        self,
        workdir: Path,
        model_name: str,
        **kwargs,
    ) -> BaseRunner:
        converted_model_path = (
            get_package_path(workdir, model_name)
            / ConvertONNX2TRT(target_precision=self.target_precision).get_output_relative_path()
        )
        trt_runner = TrtRunner(EngineFromBytes(BytesFromPath(converted_model_path.as_posix())))

        return trt_runner
