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
"""Runners global registry."""

from typing import Dict, Type, Union

import pkg_resources

from model_navigator.core.logger import LOGGER
from model_navigator.runners.base import NavigatorRunner

runner_registry: Dict[str, Type[NavigatorRunner]] = {}


def load_runners_from_entry_points():
    """Load runners from package entrypoints."""
    for entry_point in pkg_resources.iter_entry_points("model_navigator"):
        try:
            entry_point.load()
        except Exception as e:
            LOGGER.warning(f"Encoutered an error when loading entry point: {entry_point}.\n" f"Error message: {e}.")


def register_runner(runner_cls: Type[NavigatorRunner]) -> None:
    """Register runner inside the global registry.

    Args:
        runner_cls: Runner class to add to global registry

    Raises:
        ValueError when runner with the same name already exists.
    """
    if runner_cls.name() in runner_registry:
        raise ValueError(f"Runner {runner_cls.name()} already exists.")
    runner_registry[runner_cls.name()] = runner_cls
    LOGGER.debug(f"Registered runner: {runner_cls.name()}")


def get_runner(runner_name: Union[str, Type[NavigatorRunner]]) -> Type[NavigatorRunner]:
    """Return runner with given name.

    Args:
        runner_name: Name of runner that has to be returned

    Returns:
        NavigatorRunner object

    Raises:
        ValueError when runner with given name not found.
    """
    if not isinstance(runner_name, str):
        runner_name = runner_name.name()
    runner = runner_registry.get(runner_name, None)
    if runner is None:
        raise ValueError(f"Runner `{runner_name}` not available.")
    return runner
