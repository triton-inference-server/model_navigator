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
from pathlib import Path

import click

from model_navigator.cli.spec import DatasetProfileConfigCli
from model_navigator.converter.config import DatasetProfileConfig
from model_navigator.log import init_logger
from model_navigator.perf_analyzer.profiling_data import create_profiling_data
from model_navigator.utils.cli import common_options, options_from_config
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("profiling_data")


@click.command(name="create-profiling-data", help="Create profiling data in format required by Perf Analyzer")
@common_options
@click.option("-o", "--data-output-path", help="Path to output json file", type=click.Path(writable=True))
@click.option("-i", "--iterations", help="Number of samples", type=click.INT, default=128, show_default=True)
@options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@click.pass_context
def create_profiling_data_cmd(ctx, verbose: bool, data_output_path: str, iterations: int, **kwargs):
    init_logger(verbose=verbose)
    LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {kwargs.get('config_path')}")

    run_command_validators(
        ctx.command.name,
        configuration={
            "verbose": verbose,
            "data_output_path": data_output_path,
            "iterations": iterations,
            **kwargs,
        },
    )

    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)

    # obtain first defined profile in given preferences
    shapes = [
        s
        for s in [
            dataset_profile_config.opt_shapes,
            dataset_profile_config.max_shapes,
            dataset_profile_config.min_shapes,
        ]
        if s
    ]

    if not shapes:
        raise click.BadParameter("Missing defined at least single dataset profile shape")

    shapes = shapes[0]

    create_profiling_data(
        shapes=shapes,
        value_ranges=dataset_profile_config.value_ranges,
        dtypes=dataset_profile_config.dtypes,
        iterations=iterations,
        output_path=Path(data_output_path),
    )
