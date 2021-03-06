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
from typing import Optional, Tuple

import tensorflow as tf  # pytype: disable=import-error

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export import exporters
from model_navigator.framework_api.commands.export.base import ExportBase
from model_navigator.framework_api.exceptions import ExecutionContext
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class ExportTF2SavedModel(ExportBase):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Export TensorFlow2 to SavedModel",
            command_type=CommandType.EXPORT,
            target_format=Format.TF_SAVEDMODEL,
            requires=requires,
        )

    def get_output_relative_path(
        self,
    ) -> Path:
        return format_to_relative_model_path(self.target_format)

    def _get_loggers(self) -> list:
        return [tf.get_logger()]

    def __call__(
        self, model_name: str, workdir: Path, model: Optional[tf.keras.Model] = None, **kwargs
    ) -> Optional[Path]:
        LOGGER.info("TensorFlow2 to SavedModel export started")

        exported_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if exported_model_path.exists():
            LOGGER.info("Model already exists. Skipping export.")
            return self.get_output_relative_path()
        assert model is not None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        exporters.sm.get_model = lambda: model

        with ExecutionContext(exported_model_path.parent / "reproduce.py") as context:

            kwargs = {
                "exported_model_path": exported_model_path.as_posix(),
            }

            args = []
            for k, v in kwargs.items():
                s = str(v).replace("'", '"')
                args.extend([f"--{k}", s])

            context.execute_local_runtime_script(exporters.sm.__file__, exporters.sm.export, args)

        return self.get_output_relative_path()
