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

"""Set base docker images feature tests."""
import sys
from io import TextIOWrapper
from pathlib import Path

from pytest_bdd import given, scenarios, then, when  # pytype: disable=import-error
from pytest_bdd.parsers import parse  # pytype: disable=import-error

from model_navigator.results import ResultsStore, State
from model_navigator.utils import Workspace
from model_navigator.utils.process import execute_in_subprocess, execute_interactive
from model_navigator.utils.workspace import DEFAULT_WORKSPACE_PATH

scenarios(
    "features/set_docker_images.feature",
)


@given(parse("the {model_type} model with {config_type} config file"))
def the_model_with_config_file(run_context, tmpdir, artifactory_repo, model_type: str, config_type: str):
    """the {model_type} model with {config_type} config file"""

    run_context.cwd = Path(tmpdir)
    config_path = artifactory_repo.copy_model_and_config(
        model_type=model_type, config_type=config_type, to=run_context.cwd
    )
    run_context.parameters["config_path"] = config_path


@given(parse("the {parameter_name} config parameter is set to {parameter_value}"))
def the_config_parameter_is_set_to(run_context, parameter_name: str, parameter_value: str):
    """the {parameter_name} config parameter is set to {parameter_value}"""
    run_context.parameters[parameter_name] = parameter_value


@when(parse("I execute {command_name} command"))
def i_execute_command(run_context, command_name: str):
    """I execute {command_name} command."""
    cmd = ["model-navigator", command_name]
    for name, value in run_context.parameters.items():
        cmd += [f"--{name.replace('_', '-')}", str(value)]

    is_interactive = isinstance(sys.stdin, TextIOWrapper)
    if is_interactive:
        proc, output = execute_interactive(cmd, cwd=run_context.cwd)
    else:
        proc, output = execute_in_subprocess(cmd, cwd=run_context.cwd)

    run_context.cmd = " ".join(cmd)
    run_context.return_code = proc.returncode
    run_context.output = output.decode("utf-8")


@then(parse("the {command_name} subcommand results have {state} state and parameters matching:\n{parameters}"))
def the_command_results_should_have_given_state_and_parameters_matching(
    run_context, command_name: str, state: str, parameters: str
):
    """the {command_name} command results have {state} state and parameters matching:\n{parameters}"""

    workspace = Workspace(Path(run_context.cwd) / DEFAULT_WORKSPACE_PATH)
    results_store = ResultsStore(workspace)
    command_results = results_store.load(command_name.replace("-", "_"))
    expected_state = State(state.lower())
    results_states = [result.status.state for result in command_results]
    assert all(
        [actual_state == expected_state for actual_state in results_states]
    ), f"Results states: {results_states} while expecting: {expected_state}"

    parameters = parameters.splitlines() if parameters else []
    parameters = dict([tuple(parameter.split("=")) for parameter in parameters])

    def _get_parameter(parameter_name, actual_value):
        for part in parameter_name.split("."):
            if hasattr(actual_value, part):
                actual_value = getattr(actual_value, part)
            elif actual_value is not None:
                actual_value = actual_value.get(part, None)
        return actual_value

    for name, expected_value in parameters.items():
        actual_values = [_get_parameter(name, actual_value) for actual_value in command_results]
        actual_values = [
            type(expected_value)(actual_value) if expected_value is not None else actual_value
            for actual_value in actual_values
        ]
        assert all(
            [actual_value == expected_value for actual_value in actual_values]
        ), f"Actual values: {actual_values} while expecting {expected_value}"


@then(parse("the {command_name} subcommand results have {state} state"))
def the_command_results_should_have_given_state(run_context, command_name: str, state: str):
    """the {command_name} command results have {state} state"""

    workspace = Workspace(Path(run_context.cwd) / DEFAULT_WORKSPACE_PATH)
    results_store = ResultsStore(workspace)
    command_results = results_store.load(command_name.replace("-", "_"))
    expected_state = State(state.lower())
    results_states = [result.status.state for result in command_results]
    assert all(
        [actual_state == expected_state for actual_state in results_states]
    ), f"Results states: {results_states} while expecting: {expected_state}"


@then(parse("the command should {state}"))
def the_command_should_have_given_state(run_context, state: str):
    """the command should {state} ."""
    required_state = State(state.lower())
    is_interactive = isinstance(sys.stdin, TextIOWrapper)
    if not is_interactive:
        for line in run_context.output.splitlines():
            print(line)
    if required_state == State.SUCCEEDED:
        assert run_context.return_code == 0, f"the {run_context.cmd} should {state}"
    else:
        assert run_context.return_code != 0, f"the {run_context.cmd} should {state}"
