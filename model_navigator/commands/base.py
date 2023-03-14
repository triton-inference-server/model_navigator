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
"""Base module for commands common classes and helpers."""
import abc
from dataclasses import dataclass
from enum import Enum
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.runners.base import NavigatorRunner

if TYPE_CHECKING:
    from model_navigator.core.status import Status


class CommandStatus(str, Enum):
    """Status for commands."""

    OK = "OK"
    FAIL = "FAIL"
    NOOP = "NOOP"
    INITIALIZED = "INITIALIZED"
    SKIPPED = "SKIPPED"


@dataclass
class CommandOutput:
    """Command output dataclass structure."""

    status: CommandStatus
    output: Optional[Dict[str, Any]] = None
    save: bool = False


class Command(abc.ABC):
    """Base class for command definition."""

    _is_required: bool = False

    def __init_subclass__(cls, is_required: bool = False, **kwargs):
        """Initialization of a command subclass."""
        super().__init_subclass__(**kwargs)
        cls._is_required = is_required

    def __init__(self, status: Optional["Status"] = None) -> None:
        """Initialization of a command."""
        self._status = status

    def run(self, *args, **kwargs) -> CommandOutput:
        """Run command execution.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Output with command result
        """

        def _filter_dict_for_func(data_dict: Dict[str, Any], func: Callable):
            return {k: v for k, v in data_dict.items() if k in getfullargspec(func).args}

        do_execute = self._pre_run(*args, **_filter_dict_for_func(kwargs, self._pre_run))
        if do_execute:
            output = self._run(*args, **_filter_dict_for_func(kwargs, self._run))
        else:
            output = CommandOutput(CommandStatus.SKIPPED)
        self._post_run(output, *args, **_filter_dict_for_func(kwargs, self._post_run))
        return output

    def _pre_run(self, *args, **kwargs) -> bool:
        """Pre-run command execution.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            True if command should be executed, False otherwise
        """
        return True

    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> CommandOutput:
        """Run command execution.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Output with command result
        """
        raise NotImplementedError

    def _post_run(self, output: CommandOutput, *args, **kwargs) -> None:  # noqa: B027
        """Post-run command execution.

        Args:
            output: Command output
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    @classmethod
    def is_required(cls):
        """Indicates if Command should be considered as required.

        Returns:
            True if required, False otherwise
        """
        return cls._is_required

    @classmethod
    def name(cls) -> str:
        """Return name of the command.

        Return:
            Name of command as a string
        """
        return cls.__name__

    @property
    def status(self) -> Optional["Status"]:
        """Return navigator status."""
        return self._status


class ExecutionUnit:
    """Command along with configuration and runner."""

    def __init__(
        self,
        command: Type[Command],
        config: CommonConfig,
        model_config: Optional[ModelConfig] = None,
        runner_cls: Optional[Type[NavigatorRunner]] = None,
        **kwargs,
    ) -> None:
        """Initialize object.

        Args:
            command: A command to execute
            config: Global configuration provide by user
            model_config: Optional configuration of model that has to be produced by command
            runner_cls: Optional runner for correctness or performance evaluation
        """
        self.command = command
        self.config = config
        self.model_config = model_config
        self.runner_cls = runner_cls
        self.kwargs = kwargs
