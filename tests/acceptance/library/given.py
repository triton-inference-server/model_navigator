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

from pathlib import Path

from pytest_bdd import given  # pytype: disable=import-error
from pytest_bdd.parsers import parse  # pytype: disable=import-error


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
    parameter_value = parameter_value.split(" ")
    run_context.parameters[parameter_name] = parameter_value


@given(parse("removed the {parameter_name} config parameter"))
def the_config_parameter_is_unset(run_context, parameter_name: str):
    """the {parameter_name} config parameter is set to {parameter_value}"""
    if parameter_name in run_context.parameters:
        del run_context.parameters[parameter_name]
