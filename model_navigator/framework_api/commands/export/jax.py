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
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.exceptions import ExecutionContext
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import parse_kwargs_to_cmd
from model_navigator.model import Format


class ExportJAX2SavedModel(ExportBase):
    def __init__(self, jit_compile: bool, enable_xla: bool, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Export JAX to SavedModel",
            command_type=CommandType.EXPORT,
            target_format=Format.TF_SAVEDMODEL,
            requires=requires,
        )
        self.enable_xla = enable_xla
        self.jit_compile = jit_compile

    def __call__(
        self,
        model_name: str,
        model_params,
        workdir: Path,
        input_metadata: TensorMetadata,
        model: Optional[tf.keras.Model] = None,
        batch_dim: Optional[int] = 0,
        **kwargs,
    ) -> Optional[Path]:
        LOGGER.info(f"JAX to SavedModel export started {self.jit_compile=}, {self.enable_xla=}")

        exported_model_path = workdir / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return self.get_output_relative_path()
        assert model is not None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        exporters.jax2savedmodel.get_model = lambda: model
        exporters.jax2savedmodel.get_model_params = lambda: model_params

        with ExecutionContext(
            workdir=workdir,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
        ) as context:
            kwargs = {
                "exported_model_path": exported_model_path.as_posix(),
                "jit_compile": self.jit_compile,
                "enable_xla": self.enable_xla,
                "input_metadata": input_metadata.to_json(),
                "navigator_workdir": workdir.as_posix(),
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_local_runtime_script(
                exporters.jax2savedmodel.__file__, exporters.jax2savedmodel.export, args
            )

        return self.get_output_relative_path()
