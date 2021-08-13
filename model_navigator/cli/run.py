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
import logging
import shutil
from typing import List, Optional

import click

from model_navigator.cli.analyze import analyze_cmd
from model_navigator.cli.config_model_on_triton import config_model_on_triton_cmd
from model_navigator.cli.convert_model import ConversionSetConfig, convert_cmd
from model_navigator.cli.helm_chart_create import helm_chart_create_cmd
from model_navigator.cli.profile import profile_cmd
from model_navigator.cli.spec import (
    ComparatorConfigCli,
    ConversionSetConfigCli,
    DatasetProfileConfigCli,
    ModelAnalyzerAnalysisConfigCli,
    ModelAnalyzerProfileConfigCli,
    ModelAnalyzerTritonConfigCli,
    ModelConfigCli,
    ModelSignatureConfigCli,
    PerfMeasurementConfigCli,
    TritonModelSchedulerConfigCli,
)
from model_navigator.cli.triton_evaluate_model import triton_evaluate_model_cmd
from model_navigator.configurator import Configurator, log_configuration_error
from model_navigator.converter import ComparatorConfig, ConversionResult, DatasetProfileConfig
from model_navigator.device.utils import get_gpus
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.framework import SUFFIX2FRAMEWORK
from model_navigator.kubernetes.triton import TritonServer
from model_navigator.log import init_logger, log_dict
from model_navigator.model import ModelConfig, ModelSignatureConfig
from model_navigator.model_analyzer import (
    AnalyzeResult,
    ModelAnalyzerAnalysisConfig,
    ModelAnalyzerProfileConfig,
    ModelAnalyzerTritonConfig,
    ProfileResult,
    TritonLaunchMode,
)
from model_navigator.perf_analyzer import PerfMeasurementConfig
from model_navigator.results import State
from model_navigator.triton import (
    TritonClientConfig,
    TritonModelInstancesConfig,
    TritonModelSchedulerConfig,
    TritonServerConfig,
    TritonServerFactory,
)
from model_navigator.utils import Workspace, cli
from model_navigator.utils.config import dataclass2dict
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("run")


@click.command(name="run", help="Run models")
@cli.common_options
@cli.options_from_config(ModelConfig, ModelConfigCli)
@cli.options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
@cli.options_from_config(ConversionSetConfig, ConversionSetConfigCli)
@cli.options_from_config(ComparatorConfig, ComparatorConfigCli)
@cli.options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@cli.options_from_config(TritonModelSchedulerConfig, TritonModelSchedulerConfigCli)
@cli.options_from_config(ModelAnalyzerTritonConfig, ModelAnalyzerTritonConfigCli)
@cli.options_from_config(ModelAnalyzerProfileConfig, ModelAnalyzerProfileConfigCli)
@cli.options_from_config(ModelAnalyzerAnalysisConfig, ModelAnalyzerAnalysisConfigCli)
@cli.options_from_config(PerfMeasurementConfig, PerfMeasurementConfigCli)
@click.option(
    "--override-conversion-container", is_flag=True, help="Override conversion container if it already exists."
)
@click.pass_context
def run_cmd(
    ctx,
    workspace_path: str,
    override_workspace: bool,
    verbose: bool,
    gpus: List[str],
    container_version: str,
    framework_docker_image: Optional[str],
    triton_docker_image: Optional[str],
    override_conversion_container: bool,
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {kwargs.get('config_path')}")

    run_command_validators(
        ctx.command.name,
        configuration={
            "workspace_path": workspace_path,
            "override_workspace": override_workspace,
            "verbose": verbose,
            "gpus": gpus,
            "container_version": container_version,
            "framework_docker_image": framework_docker_image,
            "triton_docker_image": triton_docker_image,
            "override_conversion_container": override_conversion_container,
            **kwargs,
        },
    )

    workspace = Workspace(workspace_path)
    src_model_config = ModelConfig.from_dict(kwargs)
    src_model_signature_config = ModelSignatureConfig.from_dict(kwargs)
    conversion_set_config = ConversionSetConfig.from_dict(kwargs)
    comparator_config = ComparatorConfig.from_dict(kwargs)
    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)
    scheduler_config = TritonModelSchedulerConfig.from_dict(kwargs)
    triton_config = ModelAnalyzerTritonConfig.from_dict(kwargs)
    profile_config = ModelAnalyzerProfileConfig.from_dict(kwargs)
    analysis_config = ModelAnalyzerAnalysisConfig.from_dict(kwargs)
    perf_measurement_config = PerfMeasurementConfig.from_dict(kwargs)

    framework = SUFFIX2FRAMEWORK[src_model_config.model_path.suffix]
    framework_docker_image = framework_docker_image or framework.container_image(container_version)
    triton_docker_image = triton_docker_image or TritonServer.container_image(container_version)

    log_dict(
        "run args:",
        {
            **dataclass2dict(src_model_config),
            **dataclass2dict(conversion_set_config),
            **dataclass2dict(comparator_config),
            **dataclass2dict(src_model_signature_config),
            **dataclass2dict(dataset_profile_config),
            **dataclass2dict(scheduler_config),
            **dataclass2dict(triton_config),
            **dataclass2dict(profile_config),
            **dataclass2dict(analysis_config),
            **dataclass2dict(perf_measurement_config),
            "workspace_path": workspace_path,
            "override_workspace": override_workspace,
            "override_conversion_container": override_conversion_container,
            "framework_docker_image": framework_docker_image,
            "triton_docker_image": triton_docker_image,
            "gpus": gpus,
            "verbose": verbose,
        },
    )

    convert_results = ctx.forward(convert_cmd)
    succeeded_convert_results = [
        convert_result for convert_result in convert_results if convert_result.status.state == State.SUCCEEDED
    ]

    # gather succeeded models; they should include source model if its matches requested target_formats
    succeeded_models = [result.output_model for result in succeeded_convert_results]

    # deploy and pre-check of model correctness with perf_analyzer
    interim_model_repository = workspace.path / "interim-model-store"
    final_model_repository = workspace.path / triton_config.model_repository

    interim_model_repository.mkdir(parents=True, exist_ok=True)
    final_model_repository.mkdir(parents=True, exist_ok=True)

    results_to_analyze = []
    configurator = Configurator()

    # TODO: refactor that loop
    triton_server = _get_triton_server(
        triton_docker_image=triton_docker_image,
        gpus=gpus,
        models_repository=interim_model_repository,
        analyzer_config=triton_config,
    )
    for model_to_deploy in succeeded_models:
        LOGGER.info(f"Running triton model configuration variants generation for {model_to_deploy.name}")
        for variant in configurator.get_models_variants(model_to_deploy):
            model_to_deploy_config = ModelConfig(variant.name, model_to_deploy.path)
            error_logs = []
            try:
                LOGGER.info(f"Verifying model variant: {variant.name}")
                triton_server.start()
                triton_client = triton_server.create_grpc_client()
                triton_client_config = TritonClientConfig(server_url=triton_client.server_url)
                # other Triton related configuration are forwarded with ctx.forward
                LOGGER.info(f"Running triton model configurator for {variant.name}")
                config_result = ctx.forward(
                    config_model_on_triton_cmd,
                    **dataclass2dict(TritonModelInstancesConfig()),
                    **dataclass2dict(model_to_deploy_config),
                    **dataclass2dict(variant.optimization_config),
                    **dataclass2dict(triton_client_config),
                    model_repository=interim_model_repository,
                    load_model=True,
                )
                if config_result.status.state != State.SUCCEEDED:
                    error_logs.append(config_result.status.message)
                    continue

                LOGGER.info(f"Running triton model evaluator for {variant.name}")
                evaluate_result = ctx.forward(
                    triton_evaluate_model_cmd,
                    **dataclass2dict(triton_client_config),
                    **dataclass2dict(triton_config),
                    **dataclass2dict(dataset_profile_config),
                    **dataclass2dict(perf_measurement_config),
                    model_name=model_to_deploy_config.model_name,
                    model_version=model_to_deploy_config.model_version,
                )
                if evaluate_result.status.state != State.SUCCEEDED:
                    error_logs.append(evaluate_result.log)
                    continue

                if not error_logs:
                    LOGGER.info(f"Promoting {variant.name} to profiling.")
                    results_to_analyze.append(config_result)
            finally:
                triton_server.stop()

                if error_logs:
                    server_log = triton_server.logs()
                    LOGGER.debug(server_log)

                    log_file = log_configuration_error(
                        workspace=workspace_path,
                        model=model_to_deploy,
                        variant=variant,
                        server_log=server_log,
                        errors=error_logs,
                    )
                    LOGGER.warning(f"Unable to evaluate model {variant.name}. Logs: {log_file}")

    # move when triton server for testing purposes is shutdown
    for config_result in results_to_analyze:
        src_dir = config_result.model_dir_in_model_store
        dst_dir = final_model_repository / src_dir.name
        LOGGER.info(f"Moving model dir between model stores: {src_dir} -> {dst_dir}")
        shutil.move(src_dir.as_posix(), dst_dir.as_posix())

    if not results_to_analyze:
        LOGGER.warning(
            "No models promoted to profiling and analysis. Please, review the error logs and verify the input model."
        )
        return

    LOGGER.info("Running Model Analyzer profiling for promoted models")
    profile_result: ProfileResult = ctx.forward(
        profile_cmd,
        **dataclass2dict(triton_config),
        **dataclass2dict(profile_config),
        **dataclass2dict(perf_measurement_config),
    )
    if profile_result.status.state != State.SUCCEEDED:
        LOGGER.error(f"Model Analyzer profiling failed with message: {profile_result.status.message}")
        return

    LOGGER.info("Running Model Analyzer analysis for profiled models")
    analyze_results: List[AnalyzeResult] = ctx.forward(
        analyze_cmd,
        **dataclass2dict(analysis_config),
    )

    failed_results: List[AnalyzeResult] = [
        analyze_result for analyze_result in analyze_results if analyze_result.status.state == State.FAILED
    ]
    if failed_results:
        for result in failed_results:
            LOGGER.error(
                f"Model Analyzer analysis failed for {result.model_config_path} with message: {result.status.message}"
            )
        return

    LOGGER.info(f"Running Helm Chart generator for top {analysis_config.top_n_configs} configs")
    for analyze_result in analyze_results:
        charts_repository = workspace.path / "helm_charts"

        selected_conversion_set_config = _obtain_conversion_config(analyze_result, succeeded_convert_results)

        create_helm_chart_result = ctx.forward(
            helm_chart_create_cmd,
            charts_repository=charts_repository,
            chart_name=analyze_result.model_config_path,
            **dataclass2dict(selected_conversion_set_config),
            **dataclass2dict(analyze_result.optimization_config),
            **dataclass2dict(analyze_result.scheduler_config),
            **dataclass2dict(analyze_result.instances_config),
        )

        if create_helm_chart_result.status.state != State.SUCCEEDED:
            LOGGER.warning(f"Helm Chart generation failed with message: {create_helm_chart_result.status.message}")


def _obtain_conversion_config(
    analyze_result: AnalyzeResult, convert_results: List[ConversionResult]
) -> ConversionSetConfig:
    matching_conversion_configs = [
        convert_result.conversion_config
        for convert_result in convert_results
        if convert_result.output_model.name in analyze_result.model_name
    ]

    if len(matching_conversion_configs) > 1:
        LOGGER.debug(f"More than on conversion config match: {len(matching_conversion_configs)}. Using one with idx 0.")

    new_conversion_set_config = ConversionSetConfig.from_single_config(matching_conversion_configs[0])
    return new_conversion_set_config


def _get_triton_server(*, triton_docker_image, gpus, models_repository, analyzer_config):
    triton_config = TritonServerConfig()
    triton_config["model-repository"] = models_repository.resolve().as_posix()
    triton_config["model-control-mode"] = "explicit"
    triton_config["strict-model-config"] = "false"
    if analyzer_config.triton_launch_mode == TritonLaunchMode.LOCAL:
        triton_server = TritonServerFactory.create_server_local(
            path=analyzer_config.triton_server_path,
            config=triton_config,
        )
    elif analyzer_config.triton_launch_mode == TritonLaunchMode.DOCKER:
        triton_server = TritonServerFactory.create_server_docker(
            image=triton_docker_image,
            config=triton_config,
            gpus=get_gpus(gpus=gpus),
        )
    else:
        raise ModelNavigatorException(f"Unsupported triton_launch_mode: {analyzer_config.triton_launch_mode}")

    return triton_server
