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
import logging

import click

from model_navigator import __version__ as navigator_version
from model_navigator.cli.analyze import analyze_cmd
from model_navigator.cli.convert_model import convert_cmd
from model_navigator.cli.create_profiling_data import create_profiling_data_cmd
from model_navigator.cli.download_file import download_cmd
from model_navigator.cli.helm_chart_create import helm_chart_create_cmd
from model_navigator.cli.profile import profile_cmd
from model_navigator.cli.run import run_cmd
from model_navigator.cli.select import select_cmd
from model_navigator.cli.triton_config_model import config_model_on_triton_cmd
from model_navigator.cli.triton_evaluate_model import triton_evaluate_model_cmd

LOGGER = logging.getLogger("model-navigator")


@click.group(name="Triton Model Navigator")
@click.version_option(navigator_version)
def cli():
    pass


def main():
    cli.add_command(cmd=analyze_cmd)
    cli.add_command(cmd=profile_cmd)
    cli.add_command(cmd=convert_cmd)
    cli.add_command(cmd=create_profiling_data_cmd)
    cli.add_command(cmd=config_model_on_triton_cmd)
    cli.add_command(cmd=helm_chart_create_cmd)
    cli.add_command(cmd=run_cmd)
    cli.add_command(cmd=download_cmd)
    cli.add_command(cmd=triton_evaluate_model_cmd)
    cli.add_command(cmd=select_cmd)
    cli(max_content_width=160)


if __name__ == "__main__":
    main()
