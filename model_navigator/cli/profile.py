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

import dataclasses
import logging
import traceback
from typing import List, Optional

import click

from model_navigator.cli.create_profiling_data import create_profiling_data_cmd
from model_navigator.cli.spec import (
    DatasetProfileConfigCli,
    ModelAnalyzerProfileConfigCli,
    ModelAnalyzerTritonConfigCli,
    PerfMeasurementConfigCli,
)
from model_navigator.cli.utils import exit_cli_command, is_cli_command
from model_navigator.converter.config import DatasetProfileConfig
from model_navigator.kubernetes.triton import TritonServer
from model_navigator.log import init_logger, log_dict
from model_navigator.model_analyzer import (
    ModelAnalyzerProfileConfig,
    ModelAnalyzerTritonConfig,
    Profiler,
    ProfileResult,
)
from model_navigator.perf_analyzer import DEFAULT_RANDOM_DATA_FILENAME, PerfMeasurementConfig
from model_navigator.results import ResultsStore, State, Status
from model_navigator.utils import Workspace
from model_navigator.utils.cli import common_options, options_from_config
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("profile")


@click.command(name="profile", help="Profile models using Triton Model Analyzer")
@common_options
@options_from_config(PerfMeasurementConfig, PerfMeasurementConfigCli)
@options_from_config(ModelAnalyzerTritonConfig, ModelAnalyzerTritonConfigCli)
@options_from_config(ModelAnalyzerProfileConfig, ModelAnalyzerProfileConfigCli)
@options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@click.pass_context
def profile_cmd(
    ctx,
    verbose: bool,
    workspace_path: str,
    override_workspace: bool,
    container_version: str,
    triton_docker_image: Optional[str],
    gpus: List[str],
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {kwargs.get('config_path')}")

    run_command_validators(
        ctx.command.name,
        configuration={
            "verbose": verbose,
            "workspace_path": "workspace_path",
            "override_workspace": override_workspace,
            "container_version": container_version,
            "triton_docker_image": triton_docker_image,
            "gpus": gpus,
            **kwargs,
        },
    )

    workspace = Workspace(workspace_path)

    triton_config = ModelAnalyzerTritonConfig.from_dict(kwargs)
    profile_config = ModelAnalyzerProfileConfig.from_dict(kwargs)
    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)
    perf_measurement_config = PerfMeasurementConfig.from_dict(kwargs)

    triton_docker_image = triton_docker_image or TritonServer.container_image(container_version)

    if verbose:
        log_dict(
            "profile args:",
            {
                **dataclasses.asdict(triton_config),
                **dataclasses.asdict(profile_config),
                **dataclasses.asdict(dataset_profile_config),
                **dataclasses.asdict(perf_measurement_config),
                "workspace_path": workspace.path,
                "override_workspace": override_workspace,
                "container_version": container_version,
                "triton_docker_image": triton_docker_image,
                "gpus": gpus,
                "verbose": verbose,
            },
        )

    profiling_data_path = None
    if dataset_profile_config.value_ranges:
        profiling_data_path = workspace.path / DEFAULT_RANDOM_DATA_FILENAME
        ctx.forward(
            create_profiling_data_cmd,
            data_output_path=profiling_data_path,
            **dataclasses.asdict(dataset_profile_config),
        )

    profiler = Profiler(
        workspace=workspace,
        triton_docker_image=triton_docker_image,
        verbose=verbose,
        profile_config=profile_config,
        triton_config=triton_config,
        dataset_profile_config=dataset_profile_config,
        profiling_data_path=profiling_data_path,
        perf_measurement_config=perf_measurement_config,
        gpus=gpus,
    )

    checkpoint_path = None
    try:
        checkpoint_path = profiler.run()
        status = Status(State.SUCCEEDED, message="Model repository profiled successfully")
    except Exception:
        message = traceback.format_exc()
        LOGGER.warning(f"Encountered exception \n{message}")
        status = Status(State.FAILED, message=message)

    profile_result = ProfileResult(
        status=status,
        triton_docker_image=triton_docker_image,
        profile_config=profile_config,
        triton_config=triton_config,
        dataset_profile=dataset_profile_config,
        profiling_data_path=profiling_data_path,
        profiling_results_path=checkpoint_path,
    )

    results_store = ResultsStore(workspace)
    results_store.dump(ctx.command.name.replace("-", "_"), [profile_result])

    if is_cli_command(ctx):
        exit_cli_command(profile_result.status)

    return profile_result
