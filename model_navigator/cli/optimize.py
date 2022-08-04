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
import pathlib
import shutil
import sys
from typing import List, Optional

import click

from model_navigator.cli.analyze import analyze_cmd
from model_navigator.cli.convert_model import convert_cmd
from model_navigator.cli.helm_chart_create import helm_chart_create_cmd
from model_navigator.cli.profile import profile_cmd
from model_navigator.cli.spec import (
    ComparatorConfigCli,
    ConversionSetConfigCli,
    DatasetProfileConfigCli,
    ModelAnalyzerAnalysisConfigCli,
    ModelAnalyzerProfileConfigCli,
    ModelConfigCli,
    ModelSignatureConfigCli,
    PerfMeasurementConfigCli,
    RunTritonConfigCli,
    TensorRTCommonConfigCli,
    TritonCustomBackendParametersConfigCli,
    TritonModelInstancesConfigCli,
)
from model_navigator.cli.triton_config_model import config_model_on_triton_cmd
from model_navigator.cli.triton_evaluate_model import triton_evaluate_model_cmd
from model_navigator.common.config import BatchingConfig, TensorRTCommonConfig
from model_navigator.configurator import Configurator, TritonConfiguratorResult, log_configuration_error
from model_navigator.converter import (
    FORMAT2FRAMEWORK,
    ComparatorConfig,
    ConversionLaunchMode,
    ConversionResult,
    ConversionSetConfig,
    DatasetProfileConfig,
)
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.framework import SUFFIX2FRAMEWORK
from model_navigator.kubernetes.triton import TritonServer
from model_navigator.log import init_logger, log_dict
from model_navigator.model import Format, Model, ModelConfig, ModelSignatureConfig
from model_navigator.model_analyzer import (
    AnalyzeResult,
    ModelAnalyzerAnalysisConfig,
    ModelAnalyzerProfileConfig,
    ModelAnalyzerTritonConfig,
    ProfileResult,
    check_model_analyzer,
)
from model_navigator.perf_analyzer import PerfMeasurementConfig
from model_navigator.results import ResultsStore, State, Status
from model_navigator.triton import (
    TritonClientConfig,
    TritonModelInstancesConfig,
    TritonServerConfig,
    TritonServerFactory,
)
from model_navigator.triton.config import RunTritonConfig, TritonCustomBackendParametersConfig, TritonLaunchMode
from model_navigator.utils import Workspace, cli
from model_navigator.utils import tensorrt as tensorrt_utils
from model_navigator.utils.config import dataclass2dict
from model_navigator.utils.device import get_available_device_kinds, get_gpus
from model_navigator.utils.environment import EnvironmentStore, get_env
from model_navigator.utils.pack_workspace import pack_workspace
from model_navigator.utils.timer import Timer
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("optimize")


@click.command(name="optimize", help="Optimize models")
@cli.common_options
@cli.options_from_config(ModelConfig, ModelConfigCli)
@cli.options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
@cli.options_from_config(ConversionSetConfig, ConversionSetConfigCli)
@cli.options_from_config(ComparatorConfig, ComparatorConfigCli)
@cli.options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@cli.options_from_config(TritonCustomBackendParametersConfig, TritonCustomBackendParametersConfigCli)
@cli.options_from_config(TritonModelInstancesConfig, TritonModelInstancesConfigCli)
@cli.options_from_config(TensorRTCommonConfig, TensorRTCommonConfigCli)
@cli.options_from_config(RunTritonConfig, RunTritonConfigCli)
@cli.options_from_config(ModelAnalyzerProfileConfig, ModelAnalyzerProfileConfigCli)
@cli.options_from_config(ModelAnalyzerAnalysisConfig, ModelAnalyzerAnalysisConfigCli)
@cli.options_from_config(PerfMeasurementConfig, PerfMeasurementConfigCli)
@click.option(
    "--launch-mode",
    type=click.Choice([item.value for item in ConversionLaunchMode]),
    default=ConversionLaunchMode.DOCKER.value,
    help="The method by which to launch conversion. "
    "'local' assume conversion will be run locally. "
    "'docker' build conversion Docker and perform operations inside it.",
)
@click.option(
    "--override-conversion-container", is_flag=True, help="Override conversion container if it already exists."
)
@click.pass_context
def optimize_cmd(
    ctx,
    config_path: Optional[pathlib.Path],
    output_package: Optional[pathlib.Path],
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
    check_model_analyzer()
    timer = Timer()
    timer.start()

    init_logger(verbose=verbose)
    if config_path:
        LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {config_path}")

    configuration = {
        "workspace_path": workspace_path,
        "override_workspace": override_workspace,
        "verbose": verbose,
        "gpus": gpus,
        "container_version": container_version,
        "framework_docker_image": framework_docker_image,
        "triton_docker_image": triton_docker_image,
        "override_conversion_container": override_conversion_container,
        **kwargs,
    }
    run_command_validators(
        ctx.command.name,
        configuration=configuration,
    )

    workspace = Workspace(workspace_path)
    src_model_config = ModelConfig.from_dict(kwargs)
    src_model_signature_config = ModelSignatureConfig.from_dict(kwargs)
    conversion_set_config = ConversionSetConfig.from_dict(kwargs)
    tensorrt_common_config = TensorRTCommonConfig.from_dict(kwargs)
    comparator_config = ComparatorConfig.from_dict(kwargs)
    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)
    instances_config = TritonModelInstancesConfig.from_dict(kwargs)
    backend_config = TritonCustomBackendParametersConfig.from_dict(kwargs)
    triton_config = RunTritonConfig.from_dict(kwargs)
    profile_config = ModelAnalyzerProfileConfig.from_dict(kwargs)
    analysis_config = ModelAnalyzerAnalysisConfig.from_dict(kwargs)
    perf_measurement_config = PerfMeasurementConfig.from_dict(kwargs)

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

    arguments = {
        **dataclass2dict(src_model_config),
        **dataclass2dict(conversion_set_config),
        **dataclass2dict(tensorrt_common_config),
        **dataclass2dict(comparator_config),
        **dataclass2dict(src_model_signature_config),
        **dataclass2dict(dataset_profile_config),
        **dataclass2dict(instances_config),
        **dataclass2dict(backend_config),
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
    }
    log_dict(
        "optimize args:",
        arguments,
    )

    _update_engine_count_per_device(instances_config, profile_config)

    gpus = get_gpus(gpus)
    device_kinds = get_available_device_kinds(gpus, instances_config)
    max_batch_size = _get_max_batch_size(profile_config)

    convert_results = ctx.forward(
        convert_cmd,
        **dataclass2dict(instances_config),
        **dataclass2dict(BatchingConfig(max_batch_size=max_batch_size)),
    )
    succeeded_convert_results = [
        convert_result for convert_result in convert_results if convert_result.status.state == State.SUCCEEDED
    ]

    # gather succeeded models; they should include source model if its matches requested target_formats
    succeeded_models = [result.output_model for result in succeeded_convert_results]

    # deploy and pre-check of model correctness with perf_analyzer
    interim_model_repository = workspace.path / "interim-model-store"
    final_model_repository = workspace.path / "final-model-store"

    interim_model_repository.mkdir(parents=True, exist_ok=True)
    final_model_repository.mkdir(parents=True, exist_ok=True)

    config_results = _configure_models_on_triton(
        ctx=ctx,
        output_model_store=interim_model_repository,
        converted_models=succeeded_models,
        instances_config=instances_config,
        backend_config=backend_config,
        batching_config=BatchingConfig(max_batch_size=1),
        tensorrt_common_config=tensorrt_common_config,
        dataset_profile_config=dataset_profile_config,
        perf_measurement_config=perf_measurement_config,
        model_signature_config=src_model_signature_config,
        triton_config=triton_config,
        triton_docker_image=triton_docker_image,
        device_kinds=device_kinds,
        gpus=gpus,
        workspace=workspace,
    )

    # move when triton server for testing purposes is shutdown
    results_to_analyze = []
    for config_result in sorted(config_results, key=lambda item: item.model_config_name):
        if config_result.status.state == State.FAILED:
            LOGGER.warning(config_result.status.message)
            continue

        src_dir = config_result.model_config_path
        dst_dir = final_model_repository / src_dir.name
        LOGGER.info(f"Model {config_result.model_config_name} evaluation succeed. Moving for profiling.")
        LOGGER.debug(f"Moving model dir between model stores: {src_dir} -> {dst_dir}")
        shutil.move(src_dir.as_posix(), dst_dir.as_posix())
        results_to_analyze.append(dst_dir.name)

    if not results_to_analyze:
        sys.exit(
            "No models promoted to profiling and analysis. Please, review the error logs and verify the input model."
        )

    LOGGER.info("Running Model Analyzer profiling for promoted models")
    profile_result: ProfileResult = ctx.forward(
        profile_cmd,
        **dataclass2dict(triton_config),
        model_repository=final_model_repository,
        **dataclass2dict(profile_config),
        **dataclass2dict(perf_measurement_config),
    )
    if profile_result.status.state != State.SUCCEEDED:
        sys.exit(f"Model Analyzer profiling failed with message: {profile_result.status.message}")

    LOGGER.info("Running Model Analyzer analysis for profiled models")
    analyze_results: List[AnalyzeResult] = ctx.forward(
        analyze_cmd,
        **dataclass2dict(analysis_config),
        model_repository=final_model_repository,
    )

    failed_results: List[AnalyzeResult] = [
        analyze_result for analyze_result in analyze_results if analyze_result.status.state == State.FAILED
    ]
    if failed_results:
        for result in failed_results:
            LOGGER.error(
                f"Model Analyzer analysis failed for {result.model_config_path} with message: {result.status.message}"
            )
        sys.exit("Model Analyzer analysis failed. Please, review the log.")

    create_helm_chart_results = []
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
            **dataclass2dict(analyze_result.batching_config),
            **dataclass2dict(analyze_result.instances_config),
        )
        create_helm_chart_results.append(create_helm_chart_result)

        if create_helm_chart_result.status.state != State.SUCCEEDED:
            LOGGER.warning(f"Helm Chart generation failed with message: {create_helm_chart_result.status.message}")
    results_store = ResultsStore(workspace)
    results_store.dump("helm_chart_create", create_helm_chart_results)
    output_package_path = _get_output_package_path(src_model_config, output_package)

    timer.stop()
    pack_workspace(workspace, output_package_path, arguments, duration=timer.duration())


def _get_output_package_path(model_config, output_package):
    if output_package:
        return pathlib.Path(output_package)

    output_package_path = pathlib.Path.cwd() / f"{model_config.model_name}.triton.nav"

    LOGGER.info(f"Output package not provided. Using: {output_package_path}")

    return output_package_path


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


def _get_triton_server(*, triton_docker_image: str, gpus: List, analyzer_config: ModelAnalyzerTritonConfig):
    triton_config = TritonServerConfig()
    triton_config["model-repository"] = analyzer_config.model_repository.resolve().as_posix()
    triton_config["model-control-mode"] = "explicit"
    triton_config["strict-model-config"] = "false"

    if analyzer_config.triton_launch_mode == TritonLaunchMode.LOCAL:
        triton_server = TritonServerFactory.create_server_local(
            path=analyzer_config.triton_server_path,
            config=triton_config,
            gpus=gpus,
        )
    elif analyzer_config.triton_launch_mode == TritonLaunchMode.DOCKER:
        triton_server = TritonServerFactory.create_server_docker(
            image=triton_docker_image,
            path=analyzer_config.triton_server_path,
            config=triton_config,
            gpus=gpus,
        )
    else:
        raise ModelNavigatorException(f"Unsupported triton_launch_mode: {analyzer_config.triton_launch_mode}")

    return triton_server


def _collect_triton_environment(workspace: Workspace, triton_config: RunTritonConfig):
    if triton_config.triton_launch_mode == TritonLaunchMode.LOCAL:
        environment_info = get_env()

        environment_store = EnvironmentStore(workspace)
        environment_store.dump("configure_models_on_triton", environment_info)
    elif triton_config.triton_launch_mode == TritonLaunchMode.DOCKER:
        LOGGER.warning("Collecting environment details from Docker not implemented yet.")
    else:
        raise ModelNavigatorException(f"Unsupported triton_launch_mode: {triton_config.triton_launch_mode}")


def _configure_models_on_triton(
    ctx,
    converted_models: List,
    output_model_store: pathlib.Path,
    batching_config: BatchingConfig,
    instances_config: TritonModelInstancesConfig,
    backend_config: TritonCustomBackendParametersConfig,
    tensorrt_common_config: TensorRTCommonConfig,
    dataset_profile_config: DatasetProfileConfig,
    perf_measurement_config: PerfMeasurementConfig,
    model_signature_config: ModelSignatureConfig,
    triton_config: RunTritonConfig,
    triton_docker_image: str,
    device_kinds: List,
    gpus: List,
    workspace: Workspace,
):
    gpus = get_gpus(gpus=gpus)
    _collect_triton_environment(workspace=workspace, triton_config=triton_config)

    triton_server = _get_triton_server(
        triton_docker_image=triton_docker_image,
        gpus=gpus,
        analyzer_config=ModelAnalyzerTritonConfig.from_dict(
            {**dataclass2dict(triton_config), **{"model_repository": output_model_store}}
        ),
    )

    config_results = []

    configurator = Configurator()

    LOGGER.info("Running Triton Model Configurator for converted models")
    for model in converted_models:
        LOGGER.info(f"\t- {model.name}")

    for model_to_deploy in converted_models:
        LOGGER.info(f"Running triton model configuration variants generation for {model_to_deploy.name}")
        for variant in configurator.get_models_variants(model_to_deploy, device_kinds=device_kinds):
            LOGGER.info(f"Generated model variant {variant.name} for Triton evaluation.")
            model_to_deploy_config = ModelConfig(variant.name, model_to_deploy.path)
            model_signature_config_updated = _get_model_signature_config(model_to_deploy, model_signature_config)
            model_config_path = None
            error_logs = []
            try:
                if variant.num_required_gpus is not None and len(gpus) < variant.num_required_gpus:
                    LOGGER.warning(
                        f"  Variant {variant.name} requires {variant.num_required_gpus} gpus "
                        f"  while only {len(gpus)} is available."
                    )
                    continue

                triton_server.set_gpus(gpus[: variant.num_required_gpus])
                triton_server.start()
                triton_client = triton_server.create_grpc_client()
                triton_client_config = TritonClientConfig(server_url=triton_client.server_url)
                # other Triton related configuration are forwarded with ctx.forward
                LOGGER.debug(f"  [{variant.name}] Load on Triton")
                config_result = ctx.forward(
                    config_model_on_triton_cmd,
                    **dataclass2dict(batching_config),
                    **dataclass2dict(instances_config),
                    **dataclass2dict(model_to_deploy_config),
                    **dataclass2dict(variant.optimization_config),
                    **dataclass2dict(triton_client_config),
                    **dataclass2dict(backend_config),
                    **dataclass2dict(tensorrt_common_config),
                    model_repository=output_model_store,
                    load_model=True,
                )
                if config_result.status.state != State.SUCCEEDED:
                    error_logs.append(config_result.status.message)
                    continue

                model_config_path = pathlib.Path(config_result.model_dir_in_model_store)

                LOGGER.debug(f"  [{variant.name}] Run inference requests.")
                evaluate_result = ctx.forward(
                    triton_evaluate_model_cmd,
                    **dataclass2dict(triton_client_config),
                    **dataclass2dict(dataset_profile_config),
                    **dataclass2dict(perf_measurement_config),
                    **dataclass2dict(model_signature_config_updated),
                    model_name=model_to_deploy_config.model_name,
                    model_version=model_to_deploy_config.model_version,
                )
                if evaluate_result.status.state != State.SUCCEEDED:
                    error_logs.append(evaluate_result.log)
                    continue
            finally:
                triton_server.stop()

                if error_logs:
                    server_log = triton_server.logs()
                    LOGGER.debug(server_log)

                    log_file = log_configuration_error(
                        workspace=workspace.path,
                        model=model_to_deploy,
                        variant=variant,
                        server_log=server_log,
                        errors=error_logs,
                    )
                    status = Status(
                        state=State.FAILED,
                        log_path=log_file.as_posix(),
                        message=f"Unable to evaluate model {variant.name}. Details can be found in logfile: {log_file.absolute()}",
                    )
                else:
                    status = Status(
                        state=State.SUCCEEDED,
                        message=f"Model {variant.name} successfully loaded.",
                    )

                config_result = TritonConfiguratorResult(
                    status=status,
                    model=model_to_deploy,
                    model_config_name=variant.name,
                    model_config_path=model_config_path,
                    batching_config=batching_config,
                    backend_config=backend_config,
                    instances_config=instances_config,
                    tensorrt_common_config=tensorrt_common_config,
                    dataset_profile_config=dataset_profile_config,
                    perf_measurement_config=perf_measurement_config,
                    optimization_config=variant.optimization_config,
                    triton_config=triton_config,
                    model_signature_config=model_signature_config_updated,
                )

                config_results.append(config_result)

    results_store = ResultsStore(workspace)
    results_store.dump("configure_models_on_triton", config_results)

    return config_results


def _get_max_batch_size(profile_config: ModelAnalyzerProfileConfig):
    """Select the max batch size used for conversion and datasets based on profiling configuration"""
    if profile_config.config_search_max_batch_sizes:
        max_batch_size = max(profile_config.config_search_max_batch_sizes)
    else:
        max_batch_size = profile_config.config_search_max_batch_size

    return max_batch_size


def _update_engine_count_per_device(
    instances_config: TritonModelInstancesConfig, profile_config: ModelAnalyzerProfileConfig
):
    """Update the instance config based on passed data in config search"""
    if instances_config.engine_count_per_device:
        return

    if not profile_config.config_search_instance_counts:
        return

    for device, values in profile_config.config_search_instance_counts.items():
        instances_config.engine_count_per_device[device] = min(values)

    LOGGER.debug(f"Update engine_count_per_devices to {instances_config.engine_count_per_device}")


def _get_model_signature_config(
    model_to_deploy: Model, model_signature_config: ModelSignatureConfig
) -> ModelSignatureConfig:
    """
    Re-write the signature for TensorRT models.

    The TensorRT does not support int64/uint64 data type and is casted to int32. For correct model processing the
    input/output shapes in model signature must be re-writen to match the converted model.
    """
    signature = model_to_deploy.signature if model_to_deploy.has_signature() else model_signature_config
    if not signature:
        return ModelSignatureConfig()

    if model_to_deploy.format != Format.TENSORRT or not signature or signature.is_missing():
        LOGGER.debug(f"The format is {model_to_deploy.format} and signature {signature}. No re-write needed.")
        return signature

    LOGGER.debug(f"The format is {model_to_deploy.format}. Re-write signature.")
    return tensorrt_utils.rewrite_signature_config(signature)
