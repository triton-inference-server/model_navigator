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
import copy
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import fire

from model_navigator.framework_api.exceptions import UserError
from model_navigator.framework_api.logger import LOGGER


class FileHandlersLogging:
    """
    Disable everything that is not a FileHandler
    """

    def __init__(self):
        self._disabled_handlers = []

    def __enter__(self):
        for handler in LOGGER.handlers:
            if not isinstance(handler, logging.FileHandler):
                self._disabled_handlers.append(handler)
                LOGGER.removeHandler(handler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handler in self._disabled_handlers:
            LOGGER.addHandler(handler)

        self._disabled_handlers.clear()


class ExecutionContext(contextlib.AbstractContextManager):
    accepted_types = (int, float, bool, str)

    def __init__(
        self,
        *,
        workdir: Path,
        script_path: Optional[Path] = None,
        cmd_path: Optional[Path] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self._workdir = workdir
        self._script_path = Path(script_path) if script_path else script_path
        self._script_path = Path(script_path) if script_path else script_path
        self._cmd_path = Path(cmd_path) if cmd_path else cmd_path
        self._cache = {}
        self._output = None
        self._verbose = verbose

    def __exit__(self, exc_type, exc_value, traceback):
        if self._output is not None:
            with FileHandlersLogging():
                if not self._verbose:
                    LOGGER.info("Command output:")
                LOGGER.info(textwrap.indent(self._output, "    "))

        if exc_type is None or exc_type == UserError:
            return

        raise UserError(exc_value)

    def execute_local_runtime_script(self, path, func, args):
        shutil.copy(path, self._script_path)
        script_path_relative = self._script_path.relative_to(self._workdir)

        filtered_args = self._filter_workdir_args(args)
        cmd = self._bake_command([sys.executable, script_path_relative.as_posix()] + filtered_args)
        try:
            fire.Fire(func, args)
        except Exception as e:
            raise UserError(f"Command to reproduce error:\n{' '.join(cmd)}") from e

    def execute_external_runtime_script(self, path, args):
        shutil.copy(path, self._script_path)
        script_path_relative = self._script_path.relative_to(self._workdir)

        filtered_args = self._filter_workdir_args(args)
        cmd = [sys.executable, script_path_relative.as_posix()] + filtered_args
        self.execute_cmd(cmd)

    def execute_cmd(self, cmd, dry_run=False):
        run_cmd = self._bake_command(cmd)

        if dry_run:
            return run_cmd

        process = subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            cwd=self._workdir,
        )

        if self._verbose:
            LOGGER.info("Command output:")

        self._output = ""
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                self._output += output
                if self._verbose:
                    print(textwrap.indent(output.rstrip(), "    "))

        result = process.poll()

        if result != 0:
            raise UserError(
                f"Processes exited with error code:{result}. Command to reproduce error:\n{' '.join(run_cmd)}"
            )

    def _bake_command(self, cmd):
        LOGGER.info(f"Command: {' '.join(cmd)}")

        if self._cmd_path is None:
            raise ValueError("cmd_path is required when using `execute_cmd` method.")

        with self._cmd_path.open("w") as f:
            f.write(" ".join(cmd))

        cmd_path_relative = self._cmd_path.relative_to(self._workdir)
        run_cmd = [os.environ.get("SHELL", "bash"), cmd_path_relative.as_posix()]

        return run_cmd

    def _filter_workdir_args(self, args: List) -> List:
        filtered_args = copy.deepcopy(args)
        try:
            workdir_index = filtered_args.index("--navigator_workdir")
            del filtered_args[workdir_index : workdir_index + 2]
        except ValueError:
            pass

        return filtered_args
