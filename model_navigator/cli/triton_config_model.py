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
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from click import UsageError

from model_navigator.cli.spec import (
    ModelConfigCli,
    ModelSignatureConfigCli,
    TritonClientConfigCli,
    TritonCustomBackendParametersConfigCli,
    TritonModelInstancesConfigCli,
    TritonModelOptimizationConfigCli,
    TritonModelSchedulerConfigCli,
)
from model_navigator.cli.utils import exit_cli_command, is_cli_command
from model_navigator.exceptions import BadParameterModelNavigatorDeployerException, ModelNavigatorDeployerException
from model_navigator.log import init_logger, log_dict
from model_navigator.model import Model, ModelConfig, ModelSignatureConfig
from model_navigator.results import ResultsStore, State, Status
from model_navigator.triton import TritonClient, TritonModelStore
from model_navigator.triton.config import (
    ModelControlMode,
    TritonClientConfig,
    TritonCustomBackendParametersConfig,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
    TritonModelSchedulerConfig,
)
from model_navigator.utils import Workspace
from model_navigator.utils.cli import common_options, options_from_config
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("triton_config_model")


@dataclass
class ConfigModelResult:
    status: Status
    model_config: ModelConfig
    model_version: str
    optimization_config: TritonModelOptimizationConfig
    scheduler_config: TritonModelSchedulerConfig
    instances_config: TritonModelInstancesConfig
    model_dir_in_model_store: Optional[Path] = None


def _load_model(
    *,
    triton_client_config,
    model_name,
    model_version,
    load_model_timeout_s,
    model_control_mode: ModelControlMode,
    verbose,
):
    LOGGER.debug(f"Loading model {model_name}:{model_version} model_control_mode: {model_control_mode}")
    client = TritonClient(server_url=triton_client_config.server_url, verbose=bool(verbose))
    WAIT_FOR_SERVER_START_S = 5
    LOGGER.debug(f"Waiting for server (timeout={WAIT_FOR_SERVER_START_S}s)")
    client.wait_for_server_ready(timeout=WAIT_FOR_SERVER_START_S)
    if model_control_mode == ModelControlMode.EXPLICIT:
        LOGGER.debug("Sending load_model request")
        client.load_model(model_name=model_name)
    elif model_control_mode == ModelControlMode.POLL:
        WAIT_FOR_POLL_SCAN_S = 15
        LOGGER.debug(f"Waiting for model scans (timeout={WAIT_FOR_POLL_SCAN_S}s)")
        time.sleep(WAIT_FOR_POLL_SCAN_S)

    LOGGER.debug(f"Polling for model availability (timeout={load_model_timeout_s}s)")
    client.wait_for_model(model_name=model_name, model_version=model_version, timeout_s=load_model_timeout_s)


CMD_NAME = "triton-config-model"


@click.command(name=CMD_NAME, help="Create Triton Server model repository and model configuration")
@common_options
@options_from_config(ModelConfig, ModelConfigCli)
@click.option(
    "--model-repository",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Path to the Triton Model Repository.",
)
@click.option(
    "--load-model",
    help="Request model load on the Triton Server and ensure it is loaded.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--load-model-timeout-s",
    help="Timeout in seconds to wait until the model loads.",
    type=click.INT,
    default=100,
    show_default=True,
)
@click.option(
    "--model-control-mode",
    help="Triton Server Model Control Mode.",
    type=click.Choice([item.value for item in ModelControlMode], case_sensitive=False),
    default=ModelControlMode.EXPLICIT.value,
    show_default=True,
)
@options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
@options_from_config(TritonModelOptimizationConfig, TritonModelOptimizationConfigCli)
@options_from_config(TritonModelSchedulerConfig, TritonModelSchedulerConfigCli)
@options_from_config(TritonModelInstancesConfig, TritonModelInstancesConfigCli)
@options_from_config(TritonCustomBackendParametersConfig, TritonCustomBackendParametersConfigCli)
@options_from_config(TritonClientConfig, TritonClientConfigCli)
@click.pass_context
def config_model_on_triton_cmd(
    ctx,
    verbose: bool,
    model_version: str,
    model_repository: str,
    load_model: bool,
    load_model_timeout_s: int,
    model_control_mode: str,
    workspace_path: str,
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {kwargs.get('config_path')}")

    run_command_validators(
        ctx.command.name,
        configuration={
            "verbose": verbose,
            "model_version": model_version,
            "model_repository": model_repository,
            "load_model": load_model,
            "load_model_timeout_s": load_model_timeout_s,
            "model_control_mode": model_control_mode,
            "workspace_path": workspace_path,
            **kwargs,
        },
    )

    model_config = ModelConfig.from_dict(kwargs)
    signature_config = ModelSignatureConfig.from_dict(kwargs)
    optimization_config = TritonModelOptimizationConfig.from_dict(kwargs)
    scheduler_config = TritonModelSchedulerConfig.from_dict(kwargs)
    instances_config = TritonModelInstancesConfig.from_dict(kwargs)
    backend_parameters_config = TritonCustomBackendParametersConfig.from_dict(kwargs)
    triton_client_config = TritonClientConfig.from_dict(kwargs)
    model_control_mode = ModelControlMode(model_control_mode)

    if verbose:
        log_dict(
            f"{CMD_NAME} args:",
            {
                **dataclasses.asdict(model_config),
                **{
                    "model_version": model_version,
                    "model_repository": model_repository,
                    "load_model": load_model,
                    "load_model_timeout_s": load_model_timeout_s,
                    "model_control_mode": model_control_mode,
                },
                **dataclasses.asdict(signature_config),
                **dataclasses.asdict(optimization_config),
                **dataclasses.asdict(scheduler_config),
                **dataclasses.asdict(instances_config),
                **dataclasses.asdict(backend_parameters_config),
                **dataclasses.asdict(triton_client_config),
            },
        )

    model = Model(
        model_config.model_name,
        model_config.model_path,
        explicit_format=model_config.model_format,
        signature_if_missing=signature_config,
    )
    if verbose:
        log_dict("model:", {**dataclasses.asdict(model)})

    model_dir_in_model_store = None
    try:
        model_store = TritonModelStore(model_repository)
        model_dir_in_model_store = model_store.deploy_model(
            model=model,
            model_version=model_version,
            optimization_config=optimization_config,
            scheduler_config=scheduler_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )
        if load_model:
            _load_model(
                triton_client_config=triton_client_config,
                model_name=model_config.model_name,
                model_version=model_version,
                load_model_timeout_s=load_model_timeout_s,
                model_control_mode=model_control_mode,
                verbose=verbose,
            )
        status = Status(state=State.SUCCEEDED, message="Model configured and loaded correctly")
        exception = None
    except ModelNavigatorDeployerException as e:
        message = str(e)
        LOGGER.debug(message)
        status = Status(state=State.FAILED, message=message, log_path=e.log_path)
        exception = e
    except Exception as e:
        message = traceback.format_exc()
        LOGGER.debug(f"Encountered exception \n{message}")
        status = Status(state=State.FAILED, message=message)
        exception = e

    config_model_result = ConfigModelResult(
        status=status,
        model_config=model_config,
        model_version=model_version,
        optimization_config=optimization_config,
        scheduler_config=scheduler_config,
        instances_config=instances_config,
        model_dir_in_model_store=model_dir_in_model_store,
    )

    workspace = Workspace(workspace_path)
    results_store = ResultsStore(workspace)
    results_store.dump(ctx.command.name.replace("-", "_"), [config_model_result])

    if (
        config_model_result.status.state != State.SUCCEEDED
        and exception is not None
        and isinstance(exception, BadParameterModelNavigatorDeployerException)
    ):
        raise UsageError(config_model_result.status.message)

    if is_cli_command(ctx):
        exit_cli_command(config_model_result.status)

    return config_model_result
