# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
"""Definition of Pipeline module - Direct Acyclic Graph (DAG) of commands execution."""

import contextlib
import time
import traceback
from typing import List

from model_navigator.commands.base import CommandOutput, CommandStatus, ExecutionUnit
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.core.logger import LOGGER, LoggingContext, StdoutLogger, pad_string
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import (
    ModelNavigatorCommandNotExecutable,
    ModelNavigatorRuntimeError,
    ModelNavigatorUserInputError,
)
from model_navigator.pipelines.pipeline_context import PipelineContext
from model_navigator.reporting.optimize.events import (
    OptimizeEvent,
    default_event_emitter,
)


class Pipeline:
    """Definition of Direct Acyclic Graph (DAG) of commands execution."""

    def __init__(
        self,
        name: str,
        execution_units: List[ExecutionUnit],
    ):
        """Initialization of object.

        Args:
            name: Name of the pipeline
            execution_units: List of execution units objects
        """
        self.name = name
        self.id = name.lower().replace(" ", "_").replace("-", "_")
        self.execution_units = execution_units
        self.event_emitter = default_event_emitter()

    def run(self, workspace: Workspace, config: CommonConfig, context: PipelineContext) -> None:
        """Execute pipeline.

        Args:
            workspace: Workspace where unit is executed
            config: A global config provided by user
            context: Context of pipeline execution
        """
        LOGGER.info(pad_string(f"Pipeline {self.name!r} started"))
        self.event_emitter.emit(OptimizeEvent.PIPELINE_STARTED, name=self.name)

        for execution_unit in self.execution_units:
            command_output = self._execute_unit(
                workspace=workspace,
                execution_unit=execution_unit,
                config=config,
                context=context,
            )
            context.update(
                execution_unit=execution_unit,
                command_output=command_output,
            )
            context.save()

        self.event_emitter.emit(OptimizeEvent.PIPELINE_FINISHED)

    def _execute_unit(
        self,
        workspace: Workspace,
        execution_unit: ExecutionUnit,
        config: CommonConfig,
        context: PipelineContext,
    ) -> CommandOutput:
        """Execute a single unit.

        Args:
            workspace: Workspace where unit is executed
            execution_unit: A unit to execute
            config: Common configuration parameters
            context: Pipeline execution context

        Returns:
            Command execution result
        """
        log_dir = None
        if execution_unit.model_config:
            log_dir = workspace.path / execution_unit.model_config.path.parent

        if config.debug:
            redirect_stdout_context = StdoutLogger(LOGGER)
        else:
            redirect_stdout_context = contextlib.nullcontext()

        with (
            LoggingContext(
                log_dir=log_dir,
                command_name=execution_unit.command.name,
                runner_cls=execution_unit.runner_cls,
                model_config=execution_unit.model_config,
            ),
            redirect_stdout_context,
        ):
            start_time = time.perf_counter()
            self._emit_command_started_event(execution_unit)
            try:
                context.validate_execution(execution_unit=execution_unit)
                try:
                    LOGGER.info(pad_string(f"Command {execution_unit.command.name!r} started"))
                    input_parameters = context.command_args(
                        workspace=workspace,
                        config=config,
                        execution_unit=execution_unit,
                    )
                    command_output = execution_unit.command().run(
                        **input_parameters
                    )  # pytype: disable=not-instantiable
                except ModelNavigatorUserInputError as e:
                    command_output = CommandOutput(status=CommandStatus.FAIL)

                    if config.verbose and e.__context__:
                        LOGGER.info(e.__context__)

                    error = traceback.format_exc()
                    LOGGER.warning(
                        "Command finished with ModelNavigatorUserInputError. "
                        "The error is considered as external error. Usually caused by "
                        "incompatibilities between the model and the target formats and/or runtimes. "
                        "Please review the command output.\n"
                        f"{error}"
                    )

                except Exception:
                    command_output = CommandOutput(status=CommandStatus.FAIL)
                    error = traceback.format_exc()
                    LOGGER.error(f"Command finished with unexpected error: {error}")
            except ModelNavigatorCommandNotExecutable:
                command_output = CommandOutput(status=CommandStatus.SKIPPED)

            end_time = time.perf_counter()
            command_output.execution_time = end_time - start_time
            LOGGER.info(f"Execution time: {command_output.execution_time:.2f}[s]")

            self.emit_command_finished_event(command_output)
            if command_output.status != CommandStatus.OK and execution_unit.command.is_required():
                raise ModelNavigatorRuntimeError(
                    "The required command has failed. Please, review the log and verify the reported problems: \n"
                    f"{command_output.output}."
                )

            return command_output

    def _emit_command_started_event(self, execution_unit: ExecutionUnit):
        """Emit command started event with execution unit properties."""
        kwargs = {
            "command": execution_unit.command.name,
            "config_key": None if execution_unit.model_config is None else execution_unit.model_config.key,
            "runner_name": None if execution_unit.runner_cls is None else execution_unit.runner_cls.name(),
        }
        self.event_emitter.emit(OptimizeEvent.COMMAND_STARTED, **kwargs)

    def emit_command_finished_event(self, command_output: CommandOutput):
        """Emit command finished with status."""
        self.event_emitter.emit(OptimizeEvent.COMMAND_FINISHED, status=command_output.status)

    def __repr__(self) -> str:
        """Return pipeline name."""
        return self.name
