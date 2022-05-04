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
import csv
import dataclasses
import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from subprocess import TimeoutExpired
from typing import Dict, List, Optional

import click as click

from model_navigator.cli.create_profiling_data import create_profiling_data_cmd
from model_navigator.cli.spec import (
    DatasetProfileConfigCli,
    ModelConfigCli,
    PerfMeasurementConfigCli,
    TritonClientConfigCli,
)
from model_navigator.cli.utils import exit_cli_command, get_dataloader, is_cli_command
from model_navigator.converter import DatasetProfileConfig
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.log import log_dict, set_logger
from model_navigator.perf_analyzer import (
    DEFAULT_RANDOM_DATA_FILENAME,
    PerfAnalyzer,
    PerfAnalyzerConfig,
    PerfMeasurementConfig,
)
from model_navigator.perf_analyzer.config import SharedMemoryMode
from model_navigator.results import State, Status
from model_navigator.triton import TritonClientConfig, parse_server_url
from model_navigator.triton.utils import get_shape_params
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


@dataclass
class TritonEvaluateModelResult:
    status: Status
    log: Optional[str]


def _read_csv_file(file: str, additional_fields: Dict) -> List[Dict]:
    LOGGER.info(f"Reading data from {file}")
    rows: List[Dict] = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            avg_latency = _average_latency(row)
            row = {**additional_fields, **row, "avg latency": avg_latency}
            rows.append(row)

    LOGGER.info("done")

    return rows


def _save_csv_file(file: str, results: List[Dict]):
    LOGGER.info(f"Saving data to {file}")
    with open(file, "w") as csvfile:
        header = tuple(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    LOGGER.info("done")


def _average_latency(row: Dict):
    """
    Calculate average latency for Performance Analyzer single test
    """
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
    avg_latency = sum(int(row.get(f, 0)) for f in avg_sum_fields)

    return avg_latency


def _perf_analyzer_evaluation(
    server_url: str,
    model_name: str,
    input_shapes: List[str],
    batch_sizes: List[int],
    model_version: str = "1",
    input_data: str = "random",
    number_of_model_instances: int = 1,
    measurement_mode: MeasurementMode = MeasurementMode.COUNT_WINDOWS,
    measurement_interval: int = 5000,
    measurement_request_count: int = 50,
    concurrency_steps: int = 1,
    batching_mode: BatchingMode = BatchingMode.STATIC,
    shared_memory: SharedMemoryMode = SharedMemoryMode.NONE,
    output_shared_memory_size: int = 102400,
    latency_report_file: Optional[str] = None,
    verbose: bool = False,
    timeout: int = 600,
    bin_path: Optional[str] = None,
):
    protocol, host, port = parse_server_url(server_url)

    if batching_mode == BatchingMode.STATIC:
        max_concurrency = 1
        min_concurrency = 1
        step = 1
    elif batching_mode == BatchingMode.DYNAMIC:
        max_total_requests = 2 * max(batch_sizes) * number_of_model_instances
        max_concurrency = min(256, max_total_requests)
        step = max(1, max_concurrency // concurrency_steps)
        min_concurrency = step
        batch_sizes = [max(1, max_total_requests // 256)]
    else:
        raise ValueError(f"Unsupported batching mode: {batching_mode}")

    results = []
    output = ""
    for batch_size in batch_sizes:
        for concurrency in range(min_concurrency, max_concurrency + step, step):
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
                "concurrency-range": f"{concurrency}:{concurrency}:1",
            }

            if latency_report_file:
                params["latency-report-file"] = latency_report_file

            if verbose:
                params["verbose"] = True

            params["shared-memory"] = shared_memory.value
            params["output-shared-memory-size"] = output_shared_memory_size

            if verbose:
                log_dict(f"Perf Analyzer config for {batch_size}", params)

            perf_config = PerfAnalyzerConfig()
            for param, value in params.items():
                perf_config[param] = value

            for shape in input_shapes:
                perf_config["shape"] = shape

            perf_analyzer = PerfAnalyzer(perf_config, timeout=timeout, stream_output=verbose, bin_path=bin_path)
            perf_analyzer.run()
            output += perf_analyzer.output()

            if latency_report_file:
                partial_result = _read_csv_file(latency_report_file, additional_fields={"Batch": batch_size})
                results.extend(partial_result)

    if latency_report_file and results:
        _save_csv_file(latency_report_file, results)

    return output


@cli.common_options
@click.command(name="triton-evaluate-model", help="Evaluate model on Triton using Perf Analyzer")
@click.option("--model-name", required=True, help=ModelConfigCli.model_name.help)
@click.option("--model-version", required=False, help=ModelConfigCli.model_version.help, default="1")
@click.option(
    "--batch-sizes",
    type=str,
    required=False,
    help="List of batch sizes to tests. Comma separated integers.",
    default="1",
)
@click.option(
    "--batching-mode",
    type=click.Choice([item.value for item in BatchingMode]),
    default=BatchingMode.STATIC.value,
    help="Select model evaluation mode "
    "'static' run static batching scenario. "
    "'dynamic' run dynamic batching scenario.",
    required=False,
)
@click.option(
    "--concurrency-steps",
    type=int,
    default=16,
    help="Number of concurrency steps for dynamic batching",
    required=False,
)
@click.option(
    "--latency-report-file",
    type=str,
    required=False,
    default=None,
    help="Provide path to file where CSV report has to be stored",
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
    batch_sizes: str,
    batching_mode: str,
    verbose: bool,
    latency_report_file: str,
    concurrency_steps: int,
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

    batch_sizes = list(map(lambda v: int(v.strip()), batch_sizes.split(",")))

    if verbose:
        log_dict(
            "triton-evaluate-model args:",
            {
                **{
                    "model_name": model_name,
                    "model_version": model_version,
                    "batch_sizes": batch_sizes,
                    "concurrency_steps": concurrency_steps,
                    "latency_report_file": latency_report_file,
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
        dataloader = get_dataloader(**kwargs)
        shapes_params = get_shape_params(dataloader.max_shapes)
        if dataset_profile_config.value_ranges and dataset_profile_config.dtypes:
            profiling_data_path = workspace.path / DEFAULT_RANDOM_DATA_FILENAME
            ctx.forward(create_profiling_data_cmd, data_output_path=profiling_data_path)
            profiling_data = profiling_data_path
        elif shapes_params:
            shapes = shapes_params

        perf_analyzer_log = _perf_analyzer_evaluation(
            server_url=triton_client_config.server_url,
            model_name=model_name,
            model_version=model_version,
            batch_sizes=batch_sizes,
            input_shapes=shapes,
            input_data=profiling_data,
            measurement_mode=perf_measurement_config.perf_measurement_mode,
            measurement_interval=perf_measurement_config.perf_measurement_interval,
            measurement_request_count=perf_measurement_config.perf_measurement_request_count,
            timeout=perf_measurement_config.perf_analyzer_timeout,
            verbose=verbose,
            batching_mode=BatchingMode(batching_mode),
            shared_memory=SharedMemoryMode(perf_measurement_config.perf_measurement_shared_memory),
            output_shared_memory_size=perf_measurement_config.perf_measurement_output_shared_memory_size,
            concurrency_steps=concurrency_steps,
            latency_report_file=latency_report_file,
            bin_path=perf_measurement_config.perf_analyzer_path,
        )

        message = f"Evaluated model {model_name} and batch size {batch_sizes} and mode: {perf_measurement_config.perf_measurement_output_shared_memory_size}"
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

    if is_cli_command(ctx):
        exit_cli_command(result.status)

    return result
