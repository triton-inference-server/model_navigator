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
"""Workspace definition."""

import pathlib
import shutil
from typing import Optional, Union

from model_navigator.core.constants import DEFAULT_WORKSPACE
from model_navigator.core.logger import LOGGER, add_log_file_handler


class Workspace:
    """Workspace class to maintain shared directory to store artefact."""

    def __init__(self, path: Optional[Union[str, pathlib.Path]] = None):
        """Initialize workspace.

        If path is not provided the default workspace is used.

        Args:
            path: Path to create a workspace
        """
        if not path:
            self._path = pathlib.Path.cwd() / DEFAULT_WORKSPACE
        else:
            self._path = pathlib.Path(path)

    @property
    def path(self) -> pathlib.Path:
        """Return path to workspace."""
        return self._path

    def exists(self) -> bool:
        """Verify if workspace already exists."""
        return self._path.exists()

    def create(self) -> None:
        """Create new workspace catalog."""
        self._path.mkdir(parents=True, exist_ok=False)

    def delete(self) -> None:
        """Remove existing workspace."""
        shutil.rmtree(self._path.as_posix(), ignore_errors=False)

    def initialize(self):
        """Initializing workspace."""
        if self.exists():
            LOGGER.info(f"Removing exiting workspace at {self.path}")
            self.delete()
        assert not self.exists()

        LOGGER.info(f"Creating workspace at {self.path}")
        self.create()

        LOGGER.info("Initializing log file.")
        add_log_file_handler(log_dir=self.path)
