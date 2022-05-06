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
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

LOGGER = logging.getLogger(__name__)
DEFAULT_WORKSPACE_PATH = Path("navigator_workspace")


class Workspace:
    def __init__(self, workspace_path: Optional[Union[str, Path]] = None):
        workspace_path = workspace_path if workspace_path is not None else DEFAULT_WORKSPACE_PATH
        self._workspace_path = Path(workspace_path).resolve()
        LOGGER.debug(f"Workspace path {self._workspace_path}")
        self._workspace_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self):
        return self._workspace_path

    def exists(self):
        return self._workspace_path.exists()

    def empty(self):
        all_files = list(self.path.rglob("*"))
        if len(all_files) == 0:
            return True
        for p in all_files:
            rel_p = p.relative_to(self.path)
            if rel_p.parts and not rel_p.parts[0].startswith("."):
                return False
        return True

    def clean(self):
        LOGGER.debug(f"Cleaning workspace dir {self.path}")

        for child in self.path.rglob("*"):
            rel_p = child.relative_to(self.path)
            if len(rel_p.parts) == 0 or rel_p.parts[0].startswith("."):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        if not self.empty():
            raise OSError(f"Could not clean {self.path} workspace")
