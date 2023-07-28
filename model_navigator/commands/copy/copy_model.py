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
"""Command for copying model."""

import pathlib
import shutil
from typing import Optional

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.core.workspace import Workspace


class CopyModel(Command):
    """Copy Model command."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        model: Optional[pathlib.Path] = None,
    ) -> CommandOutput:
        """Run copy of the model.

        Args:
            workspace (Path): Model Navigator workspace path.
            path (Path): model path to copy to. Relative to workspace path.
            model (Optional[Path], optional): model path to copy from. Defaults to None.

        Returns:
            CommandOutput: Status OK.
        """
        destination_model_path = workspace.path / path
        if destination_model_path.exists():
            return CommandOutput(status=CommandStatus.SKIPPED)
        assert model is not None
        destination_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=model, dst=destination_model_path)

        return CommandOutput(status=CommandStatus.OK)
