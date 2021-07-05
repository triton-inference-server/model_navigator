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
from model_navigator.converter.config import ComparatorConfig, ConversionConfig, DatasetProfileConfig
from model_navigator.kubernetes import ChartGenerator, HelmChartGenerationResult
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
    container_version: str,
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug("Running helm_chart_create_cmd")

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
                "verbose": verbose,
            },
        )

    output_path = charts_repository / chart_name

    LOGGER.debug("Obtaining conversion config for Helm Chart")
    conversion_set_config_lst = list(conversion_set_config)
    if len(conversion_set_config_lst) == 0:
        conversion_config = ConversionConfig()
    else:
        conversion_config = conversion_set_config_lst[0]

    try:
        container_version_without_tag = container_version.split("-")[0]
        helm_chart_generator = ChartGenerator(container_version=container_version_without_tag)
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
        )
    except Exception:
        message = traceback.format_exc()
        LOGGER.debug(f"Encountered exception \n{message}")
        return HelmChartGenerationResult(
            status=Status(state=State.FAILED, message=message),
            container_version=container_version,
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
    results_store.dump("helm_chart_generation", [helm_chart_generation_result])

    return helm_chart_generation_result