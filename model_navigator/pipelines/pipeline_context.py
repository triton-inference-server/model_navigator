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
"""PipelineContext class definition."""

import collections
import dataclasses
import datetime
from typing import Any, Dict, List

import yaml
from tabulate import tabulate

from model_navigator.api.config import Format
from model_navigator.commands.base import CommandOutput, CommandStatus, ExecutionUnit
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.core.constants import NAVIGATOR_VERSION
from model_navigator.core.logger import LOGGER, pad_string
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorCommandNotExecutable, ModelNavigatorRuntimeError
from model_navigator.utils.common import DataObject
from model_navigator.utils.environment import get_env


@dataclasses.dataclass
class RunnerCommand(DataObject):
    """Runner Commands Status."""

    commands: Dict[str, CommandOutput] = dataclasses.field(default_factory=lambda: {})

    @classmethod
    def from_dict(cls, data_dict: Dict) -> "RunnerCommand":
        """Create RunnerCommand from the dictionary.

        Args:
            data_dict (Dict): dictionary with runner results data.

        Returns:
            RunnerCommand
        """
        commands = {k: CommandOutput(v) for k, v in data_dict["commands"].items()}

        return cls(
            commands=commands,
        )


@dataclasses.dataclass
class ModelCommand(DataObject):
    """Model Commands Status."""

    model_config: ModelConfig
    runners_commands: Dict[str, RunnerCommand] = dataclasses.field(default_factory=lambda: {})
    commands: Dict[str, CommandOutput] = dataclasses.field(default_factory=lambda: {})

    @classmethod
    def from_dict(cls, data_dict: Dict) -> "ModelCommand":
        """Create ModelCommand from the dictionary.

        Args:
            data_dict: dictionary with model status data.

        Returns:
            ModelCommand
        """
        return cls(
            model_config=ModelConfig.from_dict(data_dict["model_config"]),
            runners_commands={
                runner_name: RunnerCommand.from_dict(runner_res)
                for runner_name, runner_res in data_dict["runners_commands"].items()
            },
            commands={command: CommandOutput(output) for command, output in data_dict.get("commands", {}).items()},
        )


@dataclasses.dataclass
class PipelineCommands(DataObject):
    """Pipeline Commands Status."""

    models_commands: Dict[str, ModelCommand]
    commands: Dict[str, CommandOutput] = dataclasses.field(default_factory=lambda: {})

    @classmethod
    def from_dict(cls, data_dict: Dict) -> "PipelineCommands":
        """Create PipelineStatus from the dictionary.

        Args:
            data_dict: Dictionary with navigator commands data.

        Returns:
            PipelineStatus
        """
        models_commands = {
            model_key: ModelCommand.from_dict(models_commands)
            for model_key, models_commands in data_dict["models_commands"].items()
        }
        # update model_config parents
        for model_status, model_status_dict in zip(models_commands.values(), data_dict["models_commands"].values()):
            for parent_model_key in models_commands:
                if parent_model_key == model_status_dict["model_config"]["parent_key"]:
                    model_status.model_config.parent = models_commands[parent_model_key].model_config

        commands = {
            command_name: CommandOutput.from_dict(command) for command_name, command in data_dict["commands"].items()
        }

        return cls(models_commands=models_commands, commands=commands)

    def get_model_configs(self) -> Dict[Format, List[ModelConfig]]:
        """Get model configurations from the commands.

        Returns:
            Dict[Format, List[ModelConfig]]: Dictionary where key is a model format
                and value is a list of model configs for this format.
        """
        model_configs = collections.defaultdict(list)
        for models_commands in self.models_commands.values():
            model_configs[models_commands.model_config.format.value].append(models_commands.model_config)
        return model_configs


@dataclasses.dataclass
class PipelineMetadata(DataObject):
    """Metadata of pipeline execution."""

    model_navigator_version: str
    environment: Dict
    timestamp: str = dataclasses.field(default_factory=lambda: f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S.%f}")

    @classmethod
    def from_dict(cls, data_dict: Dict) -> "PipelineMetadata":
        """Create PipelineMetadata from the dictionary.

        Args:
            data_dict: Dictionary with navigator commands data.

        Returns:
            PipelineStatus
        """
        return cls(
            model_navigator_version=data_dict["model_navigator_version"],
            environment=data_dict["environment"],
            timestamp=data_dict["timestamp"],
        )


class PipelineContext:
    """PipelineContext class."""

    def __init__(self, workspace: Workspace):
        """Initialize context."""
        self._workspace = workspace
        self._file = workspace.path / "context.yaml"
        self._metadata = PipelineMetadata(
            model_navigator_version=NAVIGATOR_VERSION,
            environment=get_env(),
        )
        self._commands = PipelineCommands(models_commands={}, commands={})

    @property
    def workspace(self) -> Workspace:
        """Workspace of context."""
        return self._workspace

    @property
    def metadata(self) -> PipelineMetadata:
        """Pipeline Execution Metadata."""
        return self._metadata

    @property
    def commands(self) -> PipelineCommands:
        """Pipeline Execution Commands Status."""
        return self._commands

    def update(
        self,
        execution_unit: ExecutionUnit,
        command_output: CommandOutput,
    ) -> None:
        """Update context based on execution unit and its output.

        Args:
            execution_unit: Executed command
            command_output: command output
        """
        if execution_unit.model_config is not None:
            # If not models_commands with given model_config then add new ModelCommand.
            if execution_unit.model_config.key not in self._commands.models_commands:
                self._commands.models_commands[execution_unit.model_config.key] = ModelCommand(
                    model_config=execution_unit.model_config,
                )
            models_command = self._commands.models_commands[execution_unit.model_config.key]

            if execution_unit.runner_cls is None:
                models_command.commands[execution_unit.command.name] = command_output
            else:
                if execution_unit.runner_cls.name() not in models_command.runners_commands:
                    models_command.runners_commands[execution_unit.runner_cls.name()] = RunnerCommand()

                runners_command = models_command.runners_commands[execution_unit.runner_cls.name()]
                runners_command.commands[execution_unit.command.name] = command_output
        elif execution_unit.model_config is None:
            self._commands.commands[execution_unit.command.name] = command_output

    def initialize(self):
        """Initialize context file."""
        self._file.unlink(missing_ok=True)
        self._file.touch()

    def load(self):
        """Load context from file."""
        with self._file.open("r") as fp:
            data = yaml.safe_load(fp)

        self._commands = PipelineCommands.from_dict(data_dict=data["commands"])
        self._metadata = PipelineMetadata.from_dict(data_dict=data["metadata"])

    def save(self):
        """Save context status to file."""
        data = {
            "metadata": self._metadata.to_dict(parse=True),
            "commands": self._commands.to_dict(parse=True),
        }
        with self._file.open("w") as fp:
            yaml.safe_dump(data=data, stream=fp, sort_keys=False)

    def command_args(self, workspace: Workspace, config: CommonConfig, execution_unit: ExecutionUnit) -> Dict[str, Any]:
        """Prepare command arguments from config and current context.

        Args:
            workspace: Workspace argument passed to commands
            config: Common configuration passed execution
            execution_unit: Command with additional arguments to execute

        Return:
            Dictionary with arguments for command
        """
        input_args = {
            **config.__dict__,
            "workspace": workspace,
        }

        def _update_args(data):
            if not data:
                return

            for key, value in data.items():
                if key in input_args:
                    raise ModelNavigatorRuntimeError(f"The {key} already defined in configuration.")

                input_args[key] = value

        _update_args(data={**execution_unit.kwargs})

        if execution_unit.model_config:
            model_config_data = execution_unit.model_config.get_config_dict_for_command()
            _update_args(data=model_config_data)

            if execution_unit.runner_cls:
                _update_args(data={"runner_cls": execution_unit.runner_cls})

            model_commands = self._commands.models_commands.get(execution_unit.model_config.key)
            if model_commands:
                for model_command in model_commands.commands.values():
                    _update_args(data=model_command.output)

                if execution_unit.runner_cls:
                    runners_command = model_commands.runners_commands.get(execution_unit.runner_cls.name())
                    if runners_command:
                        for runner_command in runners_command.commands.values():
                            _update_args(data=runner_command.output)
                elif execution_unit.results_lookup_runner_cls:
                    runners_command = model_commands.runners_commands.get(
                        execution_unit.results_lookup_runner_cls.name()
                    )
                    if runners_command:
                        for runner_command in runners_command.commands.values():
                            _update_args(data=runner_command.output)

        for command_output in self._commands.commands.values():
            _update_args(data=command_output.output)

        return input_args

    def validate_execution(self, execution_unit: ExecutionUnit):
        """Validate if execution unit can be run.

        Args:
            execution_unit: An execution unit to validate

        Raises:
            ModelNavigatorCommandNotExecutable if execution unit is not executable
        """
        for required in execution_unit.command.requires():
            if execution_unit.model_config.key:
                model_command = self._commands.models_commands.get(execution_unit.model_config.key)
                if model_command is None:
                    raise ModelNavigatorCommandNotExecutable(
                        f"No command executed for model `{execution_unit.model_config.key}`."
                    )
                if execution_unit.runner_cls:
                    runner_command = model_command.runners_commands.get(execution_unit.runner_cls.name())
                    if not runner_command:
                        raise ModelNavigatorCommandNotExecutable(
                            f"""No command executed for model `{execution_unit.model_config.key}` """
                            f"""and runner `{execution_unit.runner_cls.name()}`."""
                        )
                    command = runner_command.commands.get(required)
                    if command is None:
                        raise ModelNavigatorCommandNotExecutable(f"Required command `{required}` was not executed.")
                    if command.status != CommandStatus.OK:
                        raise ModelNavigatorCommandNotExecutable(
                            f"""Status required command `{required}` for model """
                            f"""`{execution_unit.model_config.key}`  is {command.status}"""
                        )
                else:
                    command = model_command.commands.get(required)
                    if command is None:
                        raise ModelNavigatorCommandNotExecutable(f"Required command `{required}` was not executed.")
                    if command.status != CommandStatus.OK:
                        raise ModelNavigatorCommandNotExecutable(
                            f"""Status required command `{required}` for model """
                            f"""`{execution_unit.model_config.key}`  is {command.status}"""
                        )
            else:
                command = self._commands.commands.get(required)
                if command is None:
                    raise ModelNavigatorCommandNotExecutable(f"Required command `{required}` was not executed.")
                if command.status != CommandStatus.OK:
                    raise ModelNavigatorCommandNotExecutable(
                        f"Status of required command `{required}` is {command.status}"
                    )

    def log_status(self):
        """Log status."""
        summary = [[] for _ in range(len(self._commands.models_commands))]
        model_status, runner_status = None, None
        for i, model_status in enumerate(self._commands.models_commands.values()):
            summary[i].append(model_status.model_config.format.value)
            summary[i].append(model_status.model_config.key)
            if model_status.model_config.parent_key:
                summary[i].append(model_status.model_config.parent_key)
            else:
                summary[i].append("Framework")
            summary[i].append("\n".join([f"{k}: {v.status.value}" for k, v in model_status.commands.items()]))
            runtime_status = []

            for runner_name, runner_status in model_status.runners_commands.items():
                runtime_status.append(
                    "\n".join([runner_name] + [f"    {k}: {v.status.value}" for k, v in runner_status.commands.items()])
                )
            summary[i].append("\n".join(runtime_status))

        headers = ["Format", "Key", "Parent Model Key"]
        if model_status:
            headers.append("Model Status")
        if runner_status:
            headers.append("Runner Status")
        table = tabulate(summary, headers, "grid")
        LOGGER.info(f"\n{pad_string('Model Navigator Summary')}\n{table}")
