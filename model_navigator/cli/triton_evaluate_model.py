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
from dataclasses import dataclass
from enum import Enum
from subprocess import TimeoutExpired
from typing import List, Optional, Tuple, Union

import click as click

from model_navigator.cli.create_profiling_data import create_profiling_data_cmd
from model_navigator.cli.spec import (
    DatasetProfileConfigCli,
    ModelConfigCli,
    PerfMeasurementConfigCli,
    TritonClientConfigCli,
    TritonModelSchedulerConfigCli,
)
from model_navigator.converter import DatasetProfileConfig
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.log import log_dict, set_logger
from model_navigator.perf_analyzer import (
    DEFAULT_RANDOM_DATA_FILENAME,
    PerfAnalyzer,
    PerfAnalyzerConfig,
    PerfMeasurementConfig,
)
from model_navigator.results import State, Status
from model_navigator.triton import TritonClientConfig, parse_server_url
from model_navigator.utils import Workspace, cli

LOGGER = logging.getLogger("triton_evaluate_model")


class MeasurementMode(Enum):
    COUNT_WINDOWS = "count_windows"
    TIME_WINDOWS = "time_windows"


class PerformanceTool(Enum):
    """
    Available performance evaluation tools
    """

    MODEL_ANALYZER = "model_analyzer"
    PERF_ANALYZER = "perf_analyzer"


class BatchingMode(Enum):
    """
    Available batching modes
    """

    STATIC = "static"
    DYNAMIC = "dynamic"


class EvaluationMode(Enum):
    """
    Available evaluation modes
    """

    OFFLINE = "offline"
    ONLINE = "online"


class OfflineMode(Enum):
    SYSTEM = "system"
    CUDA = "cuda"


@dataclass
class TritonEvaluateModelResult:
    status: Status
    log: Optional[str]


def _perf_analyzer_evaluation(
    server_url: str,
    model_name: str,
    input_shapes: List[str],
    batch_size: int,
    model_version: str = "1",
    input_data: str = "random",
    number_of_model_instances: int = 1,
    measurement_mode: MeasurementMode = MeasurementMode.COUNT_WINDOWS,
    measurement_interval: int = 5000,
    measurement_request_count: int = 50,
    concurrency_steps: int = 1,
    batching_mode: BatchingMode = BatchingMode.STATIC,
    evaluation_mode: EvaluationMode = EvaluationMode.OFFLINE,
    offline_mode: OfflineMode = OfflineMode.SYSTEM,
    latency_report_file: Optional[str] = None,
    verbose: bool = False,
    timeout: int = 600,
):
    protocol, host, port = parse_server_url(server_url)

    if batching_mode == BatchingMode.STATIC:
        max_concurrency = 1
        min_concurrency = 1
        step = 1
    elif batching_mode == BatchingMode.DYNAMIC:
        max_total_requests = 2 * batch_size * number_of_model_instances
        max_concurrency = min(256, max_total_requests)
        step = max(1, max_concurrency // concurrency_steps)
        min_concurrency = step
        batch_size = [max(1, max_total_requests // 256)]
    else:
        raise ValueError(f"Unsupported batching mode: {batching_mode}")

    params = {
        "model-name": model_name,
        "model-version": model_version,
        "batch-size": batch_size,
        "url": f"{host}:{port}",
        "protocol": protocol,
        "input-data": input_data,
        "measurement-mode": measurement_mode,
        "measurement-request-count": measurement_request_count,
        "measurement-interval": measurement_interval,
        "concurrency-range": f"{min_concurrency}:{max_concurrency}:{step}",
    }

    if latency_report_file:
        params["latency_report_file"] = latency_report_file

    if verbose:
        params["verbose"] = True

    if evaluation_mode == EvaluationMode.OFFLINE:
        params["shared-memory"] = offline_mode.value

    if verbose:
        log_dict(f"Perf Analyzer config for {batch_size}", params)

    perf_config = PerfAnalyzerConfig()
    for param, value in params.items():
        perf_config[param] = value

    for shape in input_shapes:
        perf_config["shape"] = shape

    perf_analyzer = PerfAnalyzer(
        perf_config,
        timeout=timeout,
        stream_output=verbose,
    )
    perf_analyzer.run()
    output = perf_analyzer.output()

    return output


def _get_shape_params(dataset_profile_config):
    if not dataset_profile_config.max_shapes:
        return None

    def _shape_param_format(name, shape_):
        return f"{name}:{','.join(map(str, shape_[1:]))}"

    shape_param = [_shape_param_format(name, shape_) for name, shape_ in dataset_profile_config.max_shapes.items()]

    return shape_param


@cli.common_options
@click.command(name="triton-evaluate-model", help="Evaluate model on Triton using Perf Analyzer")
@click.option("--model-name", required=True, help=ModelConfigCli.model_name.help)
@click.option("--model-version", required=False, help=ModelConfigCli.model_version.help, default="1")
@click.option("--max-batch-size", required=False, help=TritonModelSchedulerConfigCli.max_batch_size.help, default=1)
@click.option(
    "--evaluation-mode",
    type=click.Choice([item.value for item in EvaluationMode]),
    default=EvaluationMode.OFFLINE.value,
    help="Select model evaluation mode "
    "'offline' use system or GPU memory to pass tensors. "
    "'online' use TCP to pass tensors.",
)
@click.option(
    "--offline-mode",
    type=click.Choice([item.value for item in OfflineMode]),
    default=OfflineMode.SYSTEM.value,
    help="Select offline mode "
    "'system' use system memory to pass tensors. "
    "'cuda' use GPU memory to pass tensors. ",
)
@click.option(
    "--batching-mode",
    type=click.Choice([item.value for item in BatchingMode]),
    default=BatchingMode.STATIC.value,
    help="Select model evaluation mode "
    "'static' run static batching scenario. "
    "'dynamic' run dynamic batching scenario.",
)
@cli.options_from_config(PerfMeasurementConfig, PerfMeasurementConfigCli)
@cli.options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@cli.options_from_config(TritonClientConfig, TritonClientConfigCli)
@click.pass_context
def triton_evaluate_model_cmd(
    ctx,
    workspace_path: str,
    model_name: str,
    model_version: str,
    evaluation_mode: Union[List, Tuple],
    batching_mode: str,
    offline_mode: str,
    max_batch_size: int,
    verbose: bool,
    **kwargs,
):
    """
    Evaluate model on Triton using Perf Analyzer
    """
    set_logger(verbose=verbose)

    workspace = Workspace(workspace_path)
    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)
    triton_client_config = TritonClientConfig.from_dict(kwargs)
    perf_measurement_config = PerfMeasurementConfig.from_dict(kwargs)

    if verbose:
        log_dict(
            "triton-evaluate-model args:",
            {
                **{
                    "model_name": model_name,
                    "model_version": model_version,
                    "evaluation_mode": evaluation_mode,
                    "max_batch_size": max_batch_size,
                    "verbose": verbose,
                },
                **dataclasses.asdict(perf_measurement_config),
                **dataclasses.asdict(dataset_profile_config),
                **dataclasses.asdict(triton_client_config),
            },
        )

    perf_analyzer_log = None

    profiling_data = "random"
    shapes = []

    try:
        shape_params = _get_shape_params(dataset_profile_config)
        if dataset_profile_config.value_ranges and dataset_profile_config.dtypes:
            profiling_data_path = workspace.path / DEFAULT_RANDOM_DATA_FILENAME
            ctx.forward(create_profiling_data_cmd, data_output_path=profiling_data_path)
            profiling_data = profiling_data_path
        elif shape_params:
            shapes = shape_params

        perf_analyzer_log = _perf_analyzer_evaluation(
            server_url=triton_client_config.server_url,
            model_name=model_name,
            model_version=model_version,
            batch_size=max_batch_size,
            input_shapes=shapes,
            input_data=profiling_data,
            measurement_mode=perf_measurement_config.perf_measurement_mode,
            measurement_interval=perf_measurement_config.perf_measurement_interval,
            measurement_request_count=perf_measurement_config.perf_measurement_request_count,
            timeout=perf_measurement_config.perf_analyzer_timeout,
            verbose=verbose,
            evaluation_mode=EvaluationMode(evaluation_mode),
            batching_mode=BatchingMode(batching_mode),
            offline_mode=OfflineMode(offline_mode),
        )

        message = f"Evaluated model {model_name} and batch size {max_batch_size} and modes: {','.join(evaluation_mode)}"
        result = TritonEvaluateModelResult(
            status=Status(state=State.SUCCEEDED, message=message),
            log=perf_analyzer_log,
        )
    except ModelNavigatorException as e:
        LOGGER.debug(f"Encountered exception \n{str(e)}")
        result = TritonEvaluateModelResult(
            status=Status(state=State.FAILED, message=str(e)),
            log=perf_analyzer_log,
        )
    except TimeoutExpired:
        error_message = "perf_analyzer's timeouted, verify performance measurement options."
        LOGGER.debug(error_message)
        result = TritonEvaluateModelResult(
            status=Status(state=State.FAILED, message=error_message),
            log=perf_analyzer_log,
        )
    except Exception:
        message = traceback.format_exc()
        LOGGER.debug(f"Encountered exception \n{message}")
        result = TritonEvaluateModelResult(
            status=Status(state=State.FAILED, message=message),
            log=perf_analyzer_log,
        )

    return result
