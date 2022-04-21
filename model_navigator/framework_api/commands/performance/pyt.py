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

import torch  # pytype: disable=import-error
from polygraphy.backend.base import BaseRunner

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.performance.base import PerformanceBase
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.runners.pyt import PytRunner
from model_navigator.framework_api.utils import JitType, format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class PerformanceTorchScript(PerformanceBase):
    def __init__(self, target_format: Format, target_jit_type: JitType, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Performance TorchScript",
            command_type=CommandType.PERFORMANCE,
            target_format=target_format,
            requires=requires,
        )
        self.target_jit_type = target_jit_type

    def _get_runner(
        self,
        workdir: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        model_name: str,
        target_device: str,
        **kwargs,
    ) -> BaseRunner:

        exported_model_path = get_package_path(workdir, model_name) / format_to_relative_model_path(
            format=self.target_format, jit_type=self.target_jit_type
        )
        ts_runner = PytRunner(
            torch.jit.load(exported_model_path),
            input_metadata=input_metadata,
            output_names=list(output_metadata.keys()),
            target_device=target_device,
        )

        return ts_runner
