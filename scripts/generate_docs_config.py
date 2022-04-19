#!/usr/bin/env python3
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
import pathlib
import sys
import textwrap
import typing
from collections import Counter
from enum import Enum, EnumMeta

import click

from model_navigator.cli.analyze import analyze_cmd
from model_navigator.cli.convert_model import convert_cmd
from model_navigator.cli.helm_chart_create import helm_chart_create_cmd
from model_navigator.cli.profile import profile_cmd
from model_navigator.cli.run import run_cmd
from model_navigator.cli.select import select_cmd
from model_navigator.cli.triton_config_model import config_model_on_triton_cmd
from model_navigator.utils.cli import OptionNargs

START_TAG = "START_CONFIG_LIST"
END_TAG = "END_CONFIG_LIST"


class DocCmdsEntry(typing.NamedTuple):
    doc_path: pathlib.Path
    cmds: typing.List


CATALOG = [
    DocCmdsEntry(pathlib.Path("docs/optimize_for_triton.md"), [run_cmd]),
    DocCmdsEntry(pathlib.Path("docs/advanced/conversion.md"), [convert_cmd]),
    DocCmdsEntry(pathlib.Path("docs/advanced/triton_model_configurator.md"), [config_model_on_triton_cmd]),
    DocCmdsEntry(pathlib.Path("docs/advanced/helm_charts.md"), [helm_chart_create_cmd]),
    DocCmdsEntry(pathlib.Path("docs/advanced/profiling.md"), [profile_cmd]),
    DocCmdsEntry(pathlib.Path("docs/advanced/analysis.md"), [analyze_cmd]),
    DocCmdsEntry(pathlib.Path("docs/select.md"), [select_cmd]),
]

TYPE_MAPPING = {
    "text": "str",
}


def _get_type(option):
    is_list = isinstance(option, OptionNargs)

    type_name = _get_type_name(option)

    if hasattr(option.type, "func") and isinstance(option.type.func, (Enum, EnumMeta)):
        type_str = f"choice({', '.join([str(item.value) for item in option.type.func])})"
    else:
        choices_str = f"({', '.join([str(c) for c in option.type.choices])})" if hasattr(option.type, "choices") else ""
        type_str = f"{type_name}{choices_str}"

    if is_list:
        type_str = f"list[{type_str}]"

    return type_str


def _get_type_name(option):
    try:
        type_name = {"Path": "path", "file": "path", "directory": "path"}[option.type.name]
    except KeyError:
        type_name = option.type.name

    type_name = TYPE_MAPPING.get(type_name, type_name)

    return type_name


def _generate_config_description_lines(options_with_cmds):
    _check_if_there_are_duplicated_options(options_with_cmds)
    lines = []

    options = [option for option, cmd in options_with_cmds]

    duplicated_options = {option.name: option for option in options}
    options = list(duplicated_options.values())

    for option in options:
        if option.help:
            help_lines_splatted = option.help.splitlines()
            for line in help_lines_splatted:
                help_lines = textwrap.wrap(line, width=120)
                lines.extend([f"# {help_line}" for help_line in help_lines])
        type_str = _get_type(option)
        line = f"{option.name}: {type_str}"
        if option.default:
            line += f" | default: {option.default}"
        if not option.required:
            line = f"[ {line} ]"
        lines.append(line)
        lines.append("")
    return ["```yaml"] + lines + ["```"]


def _check_if_there_are_duplicated_options(options_with_cmds):
    options = {}
    for (option, cmd) in options_with_cmds:
        opts = "-".join([opt[2:].replace("-", "_") for opt in option.opts])
        key = f"{opts}-{option.help}-{_get_type_name(option)}"
        options.setdefault(key, []).append((option, cmd))

    counter = Counter([options_and_cmds_[0][0].name for key, options_and_cmds_ in options.items()])
    duplicated_options_names = [(name, count) for name, count in counter.items() if count > 1]

    if duplicated_options_names:
        print("Duplicate options found\n")
        for name, count in duplicated_options_names:
            print("\t", name, count)
            for key, options_and_cmds_list in options.items():
                if not key.startswith(name):
                    continue
                options_and_cmds_list = list(set(options_and_cmds_list))

                for option, cmd in options_and_cmds_list:
                    print(
                        "\t\t",
                        "cmd:",
                        cmd.name,
                        "opts:",
                        option.opts,
                        "type:",
                        option.type.name,
                        "default:",
                        option.default,
                        "help:",
                        option.help,
                    )
        sys.exit(1)


def _replace_config_list(tags, doc_path, config_description_lines):
    start_tag, end_tag = tags
    with doc_path.open("r") as config_md_file:
        current_payload = config_md_file.read()
    lines = current_payload.split("\n")
    try:
        start_idx = [idx for idx, line in enumerate(lines) if start_tag in line][0]
        end_idx = [idx for idx, line in enumerate(lines) if end_tag in line][0]
        lines = lines[: start_idx + 1] + config_description_lines + lines[end_idx:]
        updated_payload = "\n".join(lines)
        with doc_path.open("w") as config_md_file:
            config_md_file.write(updated_payload)
    except IndexError:
        print(f"Could not find {tags}")


def options(params):
    return [opt for opt in params if not isinstance(opt, click.Argument)]


def main():
    options_with_cmds = [[(option, cmd) for cmd in entry.cmds for option in options(cmd.params)] for entry in CATALOG]
    for entry_options_with_cmds in options_with_cmds:
        _generate_config_description_lines(entry_options_with_cmds)

    for entry in CATALOG:
        options_with_cmds = [(option, cmd) for cmd in entry.cmds for option in options(cmd.params)]
        options_with_cmds = sorted(
            options_with_cmds, key=lambda option_and_cmd: option_and_cmd[0].required, reverse=True
        )
        config_description_lines = _generate_config_description_lines(options_with_cmds)

        tags = (START_TAG, END_TAG)
        _replace_config_list(tags, entry.doc_path, config_description_lines)


if __name__ == "__main__":
    main()
