# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Execution context module responsible for handling internal and external commands."""

import contextlib
import copy
import multiprocessing as mp
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import traceback
from typing import Callable, List, Optional, Union

import fire

from model_navigator.core.logger import LOGGER
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.utils.environment import use_multiprocessing


class ExecutionContext(contextlib.AbstractContextManager):
    """Execution Context.

    This context maintain execution of internal or external command in specified workspace.
    The context create a reproduction Python script and Bash script that allows to debug single command execution
    outside the pipeline execution.

    Example of use:
        with ExecutionContext(
            workspace=workspace,
            script_path=reproduce_script_dir / "reproduce.py",
            cmd_path=reproduce_script_dir / "reproduce.sh",
            verbose=verbose,
        ) as context:
            context.execute_python_script() # Execute python script using fire.Fire in main or child process and prepare reproduction
            context.execute_cmd() # Execute command in separate system process
    """

    accepted_types = (int, float, bool, str)

    def __init__(
        self,
        *,
        workspace: Workspace,
        script_path: Optional[pathlib.Path] = None,
        cmd_path: Optional[pathlib.Path] = None,
        verbose: bool = False,
        on_exit: Optional[Callable] = None,
    ):
        """Initialize the context.

        Args:
            workspace: directory where scripts are created and executed
            script_path: path to Python script for reproduction
            cmd_path: path to Bash script for reproduction
            verbose: enable verbose logging
            on_exit: Perform operation despite success or error
        """
        super().__init__()
        self._workspace = workspace
        self._script_path = pathlib.Path(script_path) if script_path else script_path
        self._cmd_path = pathlib.Path(cmd_path) if cmd_path else cmd_path
        self._cache = {}
        self._verbose = verbose
        self._on_exit = on_exit

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: F841
        """Exit the context and store the command output to file.

        Args:
            exc_type: class of exception
            exc_value: type of exception
            traceback: traceback of exception

        Raises:
            ModelNavigatorUserInputError when issue was cased by provided by user data
        """
        if self._on_exit is not None:
            self._on_exit()

        if exc_type is None or exc_type == ModelNavigatorUserInputError:
            return

        import traceback as tb

        message = f"{str(exc_value)}\n{''.join(tb.format_tb(traceback))}"
        raise ModelNavigatorUserInputError(message=message)

    def execute_python_script(
        self,
        path: Union[str, pathlib.Path],
        func: Callable,
        args: List,
        allow_failure: bool = False,
        run_in_isolation: bool = False,
    ):
        """Execute Python script in current runtime.

        Args:
            path: path to script that has to be used for reproduction
            func: A function that is executed by fire
            args: Additional arguments to be passed to function during execution
            allow_failure: if True, do not raise exception when script execution failed
            run_in_isolation: if True, command is run in a child process

        Note: isolation can be overridden by `use_multiprocessing`.

        Raises:
            ModelNavigatorUserInputError when command execution failed
        """
        shutil.copy(path, self._script_path)
        script_path_relative = self._script_path.relative_to(self._workspace.path)

        filtered_args = self._filter_workspace_args(args)
        cmd = self._bake_command([sys.executable, script_path_relative.as_posix()] + filtered_args)
        unwrapped_args = self._unwrap_args(args)

        if run_in_isolation and use_multiprocessing():
            child_process = mp.Process(target=self._execute_function, args=(func, unwrapped_args, allow_failure, cmd))
            child_process.start()
            child_process.join()
            if child_process.exitcode and not allow_failure:
                raise ModelNavigatorUserInputError(
                    f"Process exited with {child_process.exitcode}. Check previous logs for errors."
                )
        else:
            self._execute_function(func, unwrapped_args, allow_failure, cmd)

    def _execute_function(self, func, unwrapped_args, allow_failure, cmd):
        """Execute the given function using Fire and provided args.

        This can be run in the main or child process. For the latter the logging system
        has to be configured (as the child is spawned i.e. no logging is configured).
        """
        process_name = mp.current_process().name
        if process_name != "MainProcess":
            self._workspace.configure_logging()
            LOGGER.debug("Running command: {} in the child process: {}", cmd, process_name)

        try:
            fire.Fire(func, unwrapped_args)
        except Exception as e:
            cmd_to_reproduce_error = f"Command to reproduce error: {' '.join(cmd)}"
            if allow_failure:
                LOGGER.warning(f"Command exited with error: {traceback.format_exc()}. {cmd_to_reproduce_error}")
            else:
                if process_name != "MainProcess":
                    # we are in a child process, log error
                    LOGGER.error(f"Command exited with error: {traceback.format_exc()}. {cmd_to_reproduce_error}")
                    # prevent child process from printing exception to the stderr
                    sys.stderr = open(os.devnull, "w")

                raise ModelNavigatorUserInputError(cmd_to_reproduce_error) from e

    def execute_cmd(self, cmd: List, dry_run=False, allow_failure: bool = False):
        """Execute command as subprocess.

        Args:
            cmd: A command definition
            dry_run: If True method return prepared command, but does not execute it
            allow_failure: if True, do not raise exception when command execution failed

        Raises:
            ModelNavigatorUserInputError when command execution failed
        """
        run_cmd = self._bake_command(cmd)

        if dry_run:
            return run_cmd

        process = subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            cwd=self._workspace.path,
        )

        process_output = ""
        while True:
            output_chunk = process.stdout.readline()
            if output_chunk == "" and process.poll() is not None:
                break
            if output_chunk:
                process_output += output_chunk

        result = process.poll()

        if result != 0 or self._verbose:
            LOGGER.info("Command output:\n", textwrap.indent(process_output, "    "))

        if result != 0:
            if not allow_failure:
                raise ModelNavigatorUserInputError(
                    f"Processes exited with error code: {result}. Command to reproduce error: {' '.join(run_cmd)}"
                )
            else:
                LOGGER.warning(
                    f"Processes exited with error code: {result}. Command to reproduce error: {' '.join(run_cmd)}"
                )

    def _bake_command(self, cmd: List):
        LOGGER.info(f"Command: {' '.join(cmd)}")

        if self._cmd_path is None:
            raise ValueError("cmd_path is required when using `execute_cmd` method.")

        with self._cmd_path.open("w") as f:
            f.write(" ".join(cmd))

        cmd_path_relative = self._cmd_path.relative_to(self._workspace.path)
        run_cmd = [os.environ.get("SHELL", "bash"), cmd_path_relative.as_posix()]

        return run_cmd

    def _filter_workspace_args(self, args: List) -> List:
        filtered_args = copy.deepcopy(args)
        try:
            workspace_index = filtered_args.index("--navigator_workspace")
            del filtered_args[workspace_index : workspace_index + 2]  # noqa: E203
        except ValueError:
            pass

        return filtered_args

    def _unwrap_args(self, args: List) -> List:
        return [arg.lstrip("'").rstrip("'") for arg in args]
