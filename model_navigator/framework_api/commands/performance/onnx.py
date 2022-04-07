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
from polygraphy.backend.onnxrt import SessionFromOnnx

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.performance.base import PerformanceBase
from model_navigator.framework_api.exceptions import UserError
from model_navigator.framework_api.runners.onnx import OnnxrtRunner
from model_navigator.framework_api.utils import Framework, RuntimeProvider, get_package_path
from model_navigator.model import Format


class PerformanceONNX(PerformanceBase):
    def __init__(self, runtime_provider: RuntimeProvider, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Performance ONNX", command_type=CommandType.PERFORMANCE, target_format=Format.ONNX, requires=requires
        )
        self.runtime_provider = runtime_provider

    def _get_runner(
        self,
        model: Path,
        workdir: Path,
        framework: Framework,
        model_name: str,
        target_device: str,
        **kwargs,
    ) -> BaseRunner:
        if framework == Framework.PYT:
            from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX

            onnx_model_path = (
                get_package_path(workdir, model_name) / ExportPYT2ONNX().get_output_relative_path()
            ).as_posix()
        elif framework == Framework.TF2:
            from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2ONNX

            onnx_model_path = (
                get_package_path(workdir, model_name) / ConvertSavedModel2ONNX().get_output_relative_path()
            ).as_posix()
        elif framework == Framework.ONNX:
            # pytype: disable=attribute-error
            onnx_model_path = model.as_posix()
            # pytype: enable=attribute-error
        else:
            raise UserError(f"Unknown framework: {framework.value}")

        onnx_runner = OnnxrtRunner(SessionFromOnnx(onnx_model_path, providers=[self.runtime_provider.value]))

        return onnx_runner
