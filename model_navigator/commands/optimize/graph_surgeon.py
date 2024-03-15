# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Graph Surgeon ONNX optimization command."""

import pathlib

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.core.logger import LOGGER
from model_navigator.core.workspace import Workspace
from model_navigator.utils.common import parse_kwargs_to_cmd


class GraphSurgeonOptimize(Command):
    """Command for optimizing ONNX model with Graph Surgeon."""

    def _run(
        self,
        workspace: Workspace,
        path: pathlib.Path,
        verbose: bool,
    ) -> CommandOutput:
        """Execute command.

        Args:
            workspace: Workspace where the files are stored.
            path: Path inside the workspace where exported model is stored
            verbose: Enable verbose logging

        Returns:
            CommandOutput object with status
        """
        LOGGER.info("Graph Surgeon ONNX optimization started")
        onnx_path = workspace.path / path
        if not onnx_path.exists():
            LOGGER.info("Model not found. Skipping.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        with ExecutionContext(
            workspace=workspace,
            script_path=onnx_path.parent / "reproduce_graph_surgeon.py",
            cmd_path=onnx_path.parent / "reproduce_graph_surgeon.sh",
            verbose=verbose,
        ) as context:
            kwargs = {
                "onnx_path": onnx_path.relative_to(workspace.path).as_posix(),
            }

            args = parse_kwargs_to_cmd(kwargs)

            from model_navigator.commands.optimize import graph_surgeon_script

            context.execute_external_runtime_script(graph_surgeon_script.__file__, args)

        return CommandOutput(status=CommandStatus.OK)
