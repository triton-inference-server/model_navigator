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
import dataclasses
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from model_navigator.converter import ConversionLaunchMode
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.log import log_dict
from model_navigator.triton.config import Batching
from model_navigator.utils.env import EnvironmentState, get_environment_state

LOGGER = logging.getLogger(__name__)


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, environment_state: EnvironmentState, configuration: Dict[str, Any]):
        pass


class CommandValidator(BaseValidator):
    commands_names: List[str]


class ModelNavigatorBatchingConfiguration(CommandValidator):
    commands_names = ["run", "triton-config-model"]

    def validate(self, environment_state: EnvironmentState, configuration: Dict[str, Any]):
        max_batch_size = configuration["max_batch_size"]
        batching = configuration["batching"]

        LOGGER.debug(
            "ModelNavigatorBatchingConfiguration " f"\nbatching={batching} " f"\nmax_batch_size={max_batch_size} "
        )
        if batching != Batching.DISABLED and max_batch_size == 0:
            raise ModelNavigatorException(
                f"max_batch_size should be > 0 if batching is enabled. "
                f"\nPlease use batching={Batching.DISABLED.value} to disable batching for model."
            )


class ModelNavigatorDockerContainerShouldHaveMountedWorkspaceDir(CommandValidator):
    commands_names = ["convert"]

    def validate(self, environment_state: EnvironmentState, configuration: Dict[str, Any]):
        def _is_relative_to(a: Path, *outer):
            try:
                a.relative_to(*outer)
                return True
            except ValueError:
                return False

        workspace_path = Path(configuration["workspace_path"]).resolve()
        has_mounted_workspace_path = any(
            _is_relative_to(workspace_path, mount.container_path)
            for mount in (environment_state.docker_container_mounts or [])
        )
        is_running_in_container = environment_state.docker_container_id is not None
        launch_mode = ConversionLaunchMode(configuration.get("launch_mode"))
        is_running_converter_in_docker = launch_mode == ConversionLaunchMode.DOCKER
        LOGGER.debug(
            "ModelNavigatorDockerContainerShouldHaveMountedWorkspaceDir "
            f"is_running_in_container={is_running_in_container} "
            f"has_mounted_workspace_path={has_mounted_workspace_path} "
            f"is_running_converter_in_docker={is_running_converter_in_docker}"
        )
        if is_running_in_container and not has_mounted_workspace_path and is_running_converter_in_docker:
            raise ModelNavigatorException(
                "If running Triton Model Navigator in docker container "
                f"and have selected {ConversionLaunchMode.DOCKER} conversion launch_mode, "
                f"this container should have mounted volume containing workspace dir: {workspace_path}"
            )


def run_command_validators(command_name: str, configuration: Dict[str, Any]):
    environment_state = _get_env_state(configuration)

    command_validators = _get_command_validators(command_name)
    for validator in command_validators:
        LOGGER.debug(f"Running command validator: {validator}")
        validator.validate(environment_state=environment_state, configuration=configuration)


def _get_command_validators(command_name) -> List[CommandValidator]:
    return [
        ValidatorCls()
        for ValidatorCls in CommandValidator.__subclasses__()
        if command_name in ValidatorCls.commands_names
    ]


def _get_env_state(configuration):
    environment_state = get_environment_state()
    verbose = configuration.get("verbose", False)
    if verbose:
        log_dict("Environment state", dataclasses.asdict(environment_state))
    return environment_state
