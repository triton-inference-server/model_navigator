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

r"""
For models with variable-sized inputs you must provide the --input-shape argument so that perf_analyzer knows
what shape tensors to use. For example, for a model that has an input called IMAGE that has shape [ 3, N, M ],
where N and M are variable-size dimensions, to tell perf_analyzer to send batch-size 4 requests of shape [ 3, 224, 224 ]
`--shape IMAGE:3,224,224`.
"""
import csv
import dataclasses
import logging
import os
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import click as click
from tabulate import tabulate

from model_navigator.cli.create_profiling_data import create_profiling_data_cmd
from model_navigator.cli.spec import (
    DatasetProfileConfigCli,
    ModelConfigCli,
    TritonClientConfigCli,
    TritonModelSchedulerConfigCli,
)
from model_navigator.converter import DatasetProfileConfig
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.log import log_dict, set_logger
from model_navigator.perf_analyzer import DEFAULT_RANDOM_DATA_FILENAME, PerfAnalyzer, PerfAnalyzerConfig
from model_navigator.results import State, Status
from model_navigator.triton import TritonClientConfig, parse_server_url
from model_navigator.utils import Workspace, cli

LOGGER = logging.getLogger("triton_evaluate_model")


def _show_results(results: List[Dict]):
    headers = list(results[0].keys())
    summary = map(lambda x: list(map(lambda item: item[1], x.items())), results)
    LOGGER.debug(tabulate(summary, headers=headers))


def _calculate_average_latency(r):
    avg_sum_fields = [
        "Client Send",
        "Network+Server Send/Recv",
        "Server Queue",
        "Server Compute",
        "Server Compute Input",
        "Server Compute Infer",
        "Server Compute Output",
        "Client Recv",
    ]
    avg_latency = sum(int(r.get(f, 0)) for f in avg_sum_fields)

    return avg_latency


def _update_result(results: List, performance_file: str):
    with open(performance_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["avg latency"] = _calculate_average_latency(row)
            results.append(row)


def _dynamic_batching(
    model_name: str,
    model_version: str,
    max_batch_size: int,
    input_shapes: Optional[List[str]] = None,
    profiling_data: str = "random",
    server_url: str = "http://localhost:8000",
    measurement_window: int = 5000,
    verbose: bool = False,
) -> str:
    """
    Evaluate model with dynamic batching
    """
    LOGGER.debug("Evaluating model on Triton with Dynamic Batching")
    performance_file = "triton_performance_dynamic.csv"

    protocol, host, port = parse_server_url(server_url)
    max_total_requests = 2 * max_batch_size
    max_concurrency = min(256, max_total_requests)
    batch_size = max(1, max_total_requests // 256)

    step = max(1, max_concurrency // 16)
    min_concurrency = step

    params = {
        "model-name": model_name,
        "model-version": model_version,
        "batch-size": batch_size,
        "url": f"{host}:{port}",
        "protocol": protocol,
        "input-data": profiling_data,
        "measurement-interval": measurement_window,
        "concurrency-range": f"{min_concurrency}:{max_concurrency}:{step}",
        "verbose": verbose,
    }

    if input_shapes:
        params["shapes"] = input_shapes

    perf_config = PerfAnalyzerConfig()
    for param, value in params.items():
        perf_config[param] = value

    perf_analyzer = PerfAnalyzer(perf_config, stream_output=verbose)
    perf_analyzer.run()
    output = perf_analyzer.output()

    if os.path.isfile(performance_file):
        results = list()
        _update_result(results=results, performance_file=performance_file)
        os.remove(performance_file)

        if verbose:
            _show_results(results=results)

    return output


def _static_batching(
    model_name: str,
    model_version: str,
    max_batch_size: int,
    input_shapes: Optional[List[str]] = None,
    profiling_data: str = "random",
    server_url: str = "http://localhost:8000",
    measurement_window: int = 10000,
    verbose: bool = False,
) -> str:
    """
    Evaluate model with static batching
    """
    LOGGER.debug("Evaluating model on Triton with Static Batching")

    protocol, host, port = parse_server_url(server_url)
    results: List[Dict] = list()
    performance_file = "triton_performance_static.csv"

    max_batch_size = max_batch_size if max_batch_size > 0 else 1
    params = {
        "model-name": model_name,
        "model-version": model_version,
        "batch-size": max_batch_size,
        "url": f"{host}:{port}",
        "protocol": protocol,
        "input-data": profiling_data,
        "measurement-interval": measurement_window,
        "latency-report-file": performance_file,
        "verbose": verbose,
    }

    if input_shapes:
        params["shapes"] = input_shapes

    perf_config = PerfAnalyzerConfig()
    for param, value in params.items():
        perf_config[param] = value

    perf_analyzer = PerfAnalyzer(perf_config, stream_output=verbose)
    perf_analyzer.run()
    output = perf_analyzer.output()

    if os.path.isfile(performance_file):
        _update_result(results, performance_file)
        os.remove(performance_file)

        if verbose:
            _show_results(results=results)

    return output


class EvaluationMode(Enum):
    """
    Available evaluation modes
    """

    STATIC = "static"
    DYNAMIC = "dynamic"


def _get_shape_params(dataset_profile_config):
    if not dataset_profile_config.max_shapes:
        return None

    def _shape_param_format(name, shape_):
        return f"{name}:{','.join(map(str, shape_[1:]))}"

    shape_param = " ".join(
        [_shape_param_format(name, shape_) for name, shape_ in dataset_profile_config.max_shapes.items()]
    )
    return shape_param


@dataclass
class TritonEvaluateModelResult:
    status: Status
    static_batching_log: Optional[str]
    dynamic_batching_log: Optional[str]


@click.command(name="triton-evaluate-model", help="Evaluate model on Triton using Perf Analyzer")
@cli.common_options
@click.option("--model-name", required=True, help=ModelConfigCli.model_name.help)
@click.option("--model-version", required=False, help=ModelConfigCli.model_version.help, default="1")
@click.option("--max-batch-size", required=False, help=TritonModelSchedulerConfigCli.max_batch_size.help, default=1)
@click.option("--perf-measurement-window", required=False, default=5000)
@click.option(
    "--evaluation-mode",
    type=click.Choice([item.value for item in EvaluationMode]),
    default=[EvaluationMode.STATIC.value],
    help="Select model evaluation mode "
    "'static' run static batching scenario. "
    "'dynamic' run dynamic batching scenario.",
    multiple=True,
)
@cli.options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@cli.options_from_config(TritonClientConfig, TritonClientConfigCli)
@click.pass_context
def triton_evaluate_model_cmd(
    ctx,
    workspace_path: str,
    model_name: str,
    model_version: str,
    perf_measurement_window: int,
    evaluation_mode: Union[List, Tuple],
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

    if verbose:
        log_dict(
            "triton-evaluate-model args:",
            {
                **{
                    "model_name": model_name,
                    "model_version": model_version,
                    "measurement_window": perf_measurement_window,
                    "evaluation_mode": evaluation_mode,
                    "max_batch_size": max_batch_size,
                    "verbose": verbose,
                },
                **dataclasses.asdict(dataset_profile_config),
                **dataclasses.asdict(triton_client_config),
            },
        )

    profiling_data = "random"
    shapes = list()

    shape_params = _get_shape_params(dataset_profile_config)
    if dataset_profile_config.value_ranges:
        profiling_data_path = workspace.path / DEFAULT_RANDOM_DATA_FILENAME
        ctx.forward(create_profiling_data_cmd, data_output_path=profiling_data_path)
        profiling_data = profiling_data_path
    elif shape_params:
        shapes = shape_params

    static_batching_log = None
    dynamic_batching_log = None
    try:
        if EvaluationMode.STATIC.value in evaluation_mode:
            static_batching_log = _static_batching(
                server_url=triton_client_config.server_url,
                model_name=model_name,
                model_version=model_version,
                max_batch_size=max_batch_size,
                input_shapes=shapes,
                profiling_data=profiling_data,
                measurement_window=perf_measurement_window,
                verbose=verbose,
            )

        if EvaluationMode.DYNAMIC.value in evaluation_mode and max_batch_size > 0:
            dynamic_batching_log = _dynamic_batching(
                server_url=triton_client_config.server_url,
                model_name=model_name,
                model_version=model_version,
                input_shapes=shapes,
                profiling_data=profiling_data,
                max_batch_size=max_batch_size,
                measurement_window=perf_measurement_window,
                verbose=verbose,
            )

        message = f"Evaluated model {model_name} and batch size {max_batch_size} and modes: {','.join(evaluation_mode)}"
        result = TritonEvaluateModelResult(
            status=Status(state=State.SUCCEEDED, message=message),
            static_batching_log=static_batching_log,
            dynamic_batching_log=dynamic_batching_log,
        )
    except ModelNavigatorException as e:
        LOGGER.debug(f"Encountered exception \n{e.message}")
        result = TritonEvaluateModelResult(
            status=Status(state=State.FAILED, message=e.message),
            static_batching_log=static_batching_log,
            dynamic_batching_log=dynamic_batching_log,
        )
    except Exception:
        message = traceback.format_exc()
        LOGGER.debug(f"Encountered exception \n{message}")
        result = TritonEvaluateModelResult(
            status=Status(state=State.FAILED, message=message),
            static_batching_log=static_batching_log,
            dynamic_batching_log=dynamic_batching_log,
        )

    return result
