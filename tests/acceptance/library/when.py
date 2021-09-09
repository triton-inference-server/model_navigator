# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import sys
from io import TextIOWrapper

from pytest_bdd import when  # pytype: disable=import-error
from pytest_bdd.parsers import parse  # pytype: disable=import-error

from model_navigator.utils.process import execute_in_subprocess, execute_interactive


@when(parse("I execute {command_name} command"))
def i_execute_command(run_context, command_name: str):
    """I execute {command_name} command."""
    cmd = ["model-navigator", command_name]
    for name, value in run_context.parameters.items():
        param_name = f"--{name.replace('_', '-')}"
        if not isinstance(value, list):
            value = [value]
        param_values = list(map(str, value))

        # if value is True/False - assume it is flag argument
        if len(value) == 1 and value[0] in ["True", "False"]:
            value = value[0] == "True"
            if not value:
                continue
            else:
                param_values = []

        cmd += [param_name] + param_values

    is_interactive = isinstance(sys.stdin, TextIOWrapper)
    if is_interactive:
        proc, output = execute_interactive(cmd, cwd=run_context.cwd)
    else:
        proc, output = execute_in_subprocess(cmd, cwd=run_context.cwd)

    run_context.cmd = " ".join(cmd)
    run_context.return_code = proc.returncode
    run_context.output = output.decode("utf-8")
