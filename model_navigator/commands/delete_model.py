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
"""Command for deleting models."""

import os
import shutil
from pathlib import Path

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.core.logger import LOGGER
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorRuntimeError


class DeleteModel(Command):
    """Delete model command."""

    def _run(
        self,
        workspace: Workspace,
        path: Path,
    ) -> CommandOutput:
        """Run delete model command.

        Args:
            workspace (Path): Model Navigator workspace path.
            path (Path): model path to be deleted. Relative to workspace path.

        Returns:
            CommandOutput: Status OK.
        """
        model_path = workspace.path / path
        if model_path.is_file():
            os.unlink(model_path)
            LOGGER.info(f"Model file {model_path} deleted.")
        elif model_path.is_dir():
            shutil.rmtree(model_path)
            LOGGER.info(f"Model catalog {model_path} deleted.")
        else:
            raise ModelNavigatorRuntimeError(f"File or catalog {model_path} not found.")

        return CommandOutput(status=CommandStatus.OK)
