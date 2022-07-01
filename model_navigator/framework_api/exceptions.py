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


import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import fire

from model_navigator.framework_api.logger import LOGGER


class UserError(Exception):
    pass


class ExecutionContext(contextlib.AbstractContextManager):
    accepted_types = (int, float, bool, str)

    def __init__(self, path: Optional[Path] = None):
        super().__init__()
        self._path = Path(path) if path else path
        self._cache = {}
        self._output = None

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None or exc_type == UserError:
            return

        raise UserError(exc_value)

    def execute_local_runtime_script(self, path, func, args):
        shutil.copy(path, self._path)
        try:
            fire.Fire(func, args)
        except Exception as e:
            raise UserError(
                f"Command to reproduce error:\n{' '.join([sys.executable, self._path.as_posix()] + args)}"
            ) from e
        self._path.unlink()

    def execute_external_runtime_script(self, path, args):
        shutil.copy(path, self._path)
        cmd = [sys.executable, self._path.as_posix()] + args
        self.execute_cmd(cmd)

    def execute_cmd(self, cmd):
        output = subprocess.run(cmd, capture_output=True)

        LOGGER.info(f"Command args: {output.args}")
        LOGGER.info(f"Command output: {output.stdout.decode('utf-8')}")
        if output.returncode != 0:
            raise UserError(f"{output.stderr.decode('utf-8')}\nCommand to reproduce error:\n{' '.join(cmd)}")
        if self._path and self._path.exists():
            self._path.unlink()


class TensorTypeError(TypeError):
    pass
