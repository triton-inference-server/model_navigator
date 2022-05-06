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


import shutil
from pathlib import Path
from typing import Optional, Tuple

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.utils import format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class CopyONNX(Command):
    def __init__(
        self,
        requires: Tuple[Command, ...] = (),
    ):
        super().__init__(
            name="Copy ONNX model to Navigator package",
            command_type=CommandType.COPY,
            target_format=Format.ONNX,
            requires=requires,
        )

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format)

    def __call__(
        self,
        model: Path,
        workdir: Path,
        model_name: str,
        **kwargs,
    ) -> Optional[Path]:
        destination_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        destination_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=model, dst=destination_model_path)
        return self.get_output_relative_path()
