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

import json
import re
import sys
from io import TextIOWrapper
from pathlib import Path

from pytest_bdd import then  # pytype: disable=import-error
from pytest_bdd.parsers import parse  # pytype: disable=import-error

from model_navigator.results import ResultsStore, State
from model_navigator.utils import Workspace
from model_navigator.utils.workspace import DEFAULT_WORKSPACE_PATH
from tests.utils.analyze_results import get_analyze_results
from tests.utils.profile_configs import ModelConfig, get_profiling_configs, parse_expected_configs_jsonlines
from tests.utils.triton_model_config import equal_model_configs_sets


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


@then(parse("the {model_name} model configs in latest profiling are\n{expected_configs_jsonlines}"))
def the_model_configs_in_latest_profile_checkpoint_are(run_context, model_name: str, expected_configs_jsonlines: str):
    def _filter_out_not_swappable_parameters(config):
        for name in ["name", "platform", "backend", "version_policy", "input", "output"]:
            if name in config:
                del config[name]
        return config

    expected_configs_jsonlines = parse_expected_configs_jsonlines(expected_configs_jsonlines)
    expected_configs = [json.loads(line) for line in expected_configs_jsonlines.splitlines()]

    workspace = Workspace(Path(run_context.cwd) / DEFAULT_WORKSPACE_PATH)
    profiled_configs = get_profiling_configs(workspace)
    profiled_configs = [config for name, config in profiled_configs.items() if name.startswith(model_name)]
    profiled_configs = [_filter_out_not_swappable_parameters(config) for config in profiled_configs]
    if not equal_model_configs_sets(profiled_configs, expected_configs):
        print("Profiled configs")
        for profiled_config in profiled_configs:
            print(json.dumps(profiled_config))

        print("Expected configs")
        for expected_config in expected_configs:
            print(json.dumps(expected_config))

    assert equal_model_configs_sets(profiled_configs, expected_configs)


@then(parse("the {model_name} model config in {model_repository} is equal to\n{expected_config_jsonline}"))
def the_model_configs_is_equal_to(run_context, model_name: str, model_repository: str, expected_config_jsonline: str):
    model_dir_path = Path(run_context.cwd) / model_repository / model_name
    expected_config = json.loads(expected_config_jsonline)
    created_config = ModelConfig.create_from_file(model_dir_path).to_dict()

    if expected_config != created_config:
        print("Created config")
        print(json.dumps(created_config))

        print("Expected configs")
        print(json.dumps(expected_config))

    assert expected_config == created_config


@then(parse("the {model_name} model was profiled with {concurrency_levels} concurrency levels"))
def the_concurrency_in_latest_profile_are(run_context, model_name: str, concurrency_levels: str):
    expected_concurrency = set(map(int, concurrency_levels.split(" ")))

    workspace = Workspace(Path(run_context.cwd) / DEFAULT_WORKSPACE_PATH)
    analyze_results = get_analyze_results(workspace)

    used_concurrency = {int(result["Concurrency"]) for result in analyze_results if result["Model"] == model_name}
    if used_concurrency != expected_concurrency:
        print("Used concurrency")
        print(used_concurrency)

        print("Expected concurrency")
        print(expected_concurrency)

    assert used_concurrency == expected_concurrency


@then(parse("the {pattern} pattern is present on command output"))
def substring_is_present_on_stderr(run_context, pattern: str):
    fragments_found = re.findall(pattern, run_context.output, re.IGNORECASE)
    if not fragments_found:
        print(f"Searching for pattern: {pattern}")
        print("Command output:")
        for line in run_context.output.splitlines():
            print(line)
    assert fragments_found


@then(parse("the {file_or_dir} exists"))
def the_file_exists(run_context, file_or_dir: str):
    """the {file} exists"""
    dst_path = Path(run_context.cwd) / file_or_dir
    assert dst_path.is_file() or dst_path.is_dir()
