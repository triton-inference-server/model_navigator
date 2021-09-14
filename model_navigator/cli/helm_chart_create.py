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
from pathlib import Path
from typing import Optional

import click

from model_navigator.cli.convert_model import ConversionSetConfig
from model_navigator.cli.spec import (
    ComparatorConfigCli,
    ConversionSetHelmChartConfigCli,
    DatasetProfileConfigCli,
    ModelConfigCli,
    ModelSignatureConfigCli,
    TritonModelInstancesConfigCli,
    TritonModelOptimizationConfigCli,
    TritonModelSchedulerConfigCli,
)
from model_navigator.converter.config import ComparatorConfig, DatasetProfileConfig
from model_navigator.converter.utils import FORMAT2FRAMEWORK
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.framework import SUFFIX2FRAMEWORK
from model_navigator.kubernetes import ChartGenerator, HelmChartGenerationResult
from model_navigator.kubernetes.triton import TritonServer
from model_navigator.log import init_logger, log_dict
from model_navigator.model import ModelConfig, ModelSignatureConfig
from model_navigator.results import ResultsStore, State, Status
from model_navigator.triton.config import (
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
    TritonModelSchedulerConfig,
)
from model_navigator.utils.cli import common_options, options_from_config
from model_navigator.utils.workspace import Workspace
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("helm_chart_create")


@click.command(name="helm-chart-create", help="Create helm chart for given configuration")
@common_options
@options_from_config(ModelConfig, ModelConfigCli)
@click.option(
    "--charts-repository",
    type=click.Path(writable=True),
    required=True,
    help="Path to Helm Charts repository.",
)
@click.option("--chart-name", required=True, help="Name of the chart in Helm Charts repository.")
@click.option("--chart-version", required=False, help="Version of the chart in Helm Charts repository.")
@options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
@options_from_config(ConversionSetConfig, ConversionSetHelmChartConfigCli)
@options_from_config(ComparatorConfig, ComparatorConfigCli)
@options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@options_from_config(TritonModelOptimizationConfig, TritonModelOptimizationConfigCli)
@options_from_config(TritonModelSchedulerConfig, TritonModelSchedulerConfigCli)
@options_from_config(TritonModelInstancesConfig, TritonModelInstancesConfigCli)
@click.pass_context
def helm_chart_create_cmd(
    ctx,
    verbose: bool,
    workspace_path: str,
    override_workspace: bool,
    charts_repository: str,
    chart_name: str,
    chart_version: Optional[str],
    container_version: str,
    triton_docker_image: Optional[str],
    framework_docker_image: Optional[str],
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {kwargs.get('config_path')}")

    run_command_validators(
        ctx.command.name,
        configuration={
            "verbose": verbose,
            "workspace_path": workspace_path,
            "override_workspace": override_workspace,
            "charts_repository": charts_repository,
            "chart_name": chart_name,
            "chart_version": chart_version,
            "container_version": container_version,
            "triton_docker_image": triton_docker_image,
            "framework_docker_image": framework_docker_image,
            **kwargs,
        },
    )

    workspace = Workspace(workspace_path)
    charts_repository = Path(charts_repository).resolve()

    src_model_config = ModelConfig.from_dict(kwargs)
    src_model_signature_config = ModelSignatureConfig.from_dict(kwargs)
    conversion_set_config = ConversionSetConfig.from_dict(kwargs)
    comparator_config = ComparatorConfig.from_dict(kwargs)
    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)
    optimization_config = TritonModelOptimizationConfig.from_dict(kwargs)
    scheduler_config = TritonModelSchedulerConfig.from_dict(kwargs)
    instances_config = TritonModelInstancesConfig.from_dict(kwargs)

    if src_model_config.model_format:
        framework = FORMAT2FRAMEWORK[src_model_config.model_format]
    elif src_model_config.model_path.suffix:
        framework = SUFFIX2FRAMEWORK[src_model_config.model_path.suffix]
    else:
        raise ModelNavigatorException(
            """The model format or file/directory suffix is required. Provided: \n"""
            f"""model-format: {src_model_config.model_format}\n"""
            f"""model-path: {src_model_config.model_path}"""
        )

    framework_docker_image = framework_docker_image or framework.container_image(container_version)
    triton_docker_image = triton_docker_image or TritonServer.container_image(container_version)

    if verbose:
        log_dict(
            "helm_chart_create_cmd args:",
            {
                **dataclasses.asdict(src_model_config),
                **dataclasses.asdict(src_model_signature_config),
                **dataclasses.asdict(conversion_set_config),
                **dataclasses.asdict(comparator_config),
                **dataclasses.asdict(dataset_profile_config),
                **dataclasses.asdict(optimization_config),
                **dataclasses.asdict(scheduler_config),
                **dataclasses.asdict(instances_config),
                "charts_repository": charts_repository,
                "chart_name": chart_name,
                "workspace_path": workspace.path,
                "override_workspace": override_workspace,
                "container_version": container_version,
                "framework_docker_image": framework_docker_image,
                "triton_docker_image": triton_docker_image,
                "verbose": verbose,
            },
        )

    output_path = charts_repository / chart_name

    LOGGER.debug("Obtaining conversion config for Helm Chart")
    conversion_set_config_lst = list(conversion_set_config)
    conversion_config = conversion_set_config_lst[0]

    try:
        helm_chart_generator = ChartGenerator(
            triton_docker_image=triton_docker_image, framework_docker_image=framework_docker_image
        )
        helm_chart_generation_result = helm_chart_generator.run(
            src_model=src_model_config,
            src_model_signature_config=src_model_signature_config,
            conversion_config=conversion_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
            optimization_config=optimization_config,
            scheduler_config=scheduler_config,
            instances_config=instances_config,
            output_path=output_path,
            chart_version=chart_version,
            framework=framework,
        )
    except Exception:
        message = traceback.format_exc()
        LOGGER.debug(f"Encountered exception \n{message}")
        return HelmChartGenerationResult(
            status=Status(state=State.FAILED, message=message),
            triton_docker_image=triton_docker_image,
            framework_docker_image=framework_docker_image,
            src_model_config=src_model_config,
            src_model_signature_config=src_model_signature_config,
            conversion_config=conversion_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
            optimization_config=optimization_config,
            scheduler_config=scheduler_config,
            instances_config=instances_config,
            helm_chart_dir_path=None,
        )

    results_store = ResultsStore(workspace)
    results_store.dump(ctx.command.name.replace("-", "_"), [helm_chart_generation_result])

    return helm_chart_generation_result
