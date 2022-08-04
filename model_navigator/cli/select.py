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
import io
import logging
import pathlib
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import click

from model_navigator.cli.spec import SelectConfigCli
from model_navigator.converter.config import TRITON_SUPPORTED_FORMATS, TensorRTPrecision
from model_navigator.log import init_logger, log_dict
from model_navigator.model import Format
from model_navigator.model_analyzer import ModelAnalyzerAnalysisConfig
from model_navigator.model_analyzer.summary import Summary
from model_navigator.triton.config import BackendAccelerator
from model_navigator.utils import cli
from model_navigator.utils.config import BaseConfig, dataclass2dict
from model_navigator.utils.triton_package import TritonPackage

LOGGER = logging.getLogger(__name__)


@dataclass
class SelectConfig(BaseConfig):
    objective: Dict[str, int] = field(default_factory=lambda: {"perf_throughput": 10})
    max_latency_ms: Optional[int] = None
    min_throughput: int = 0
    max_gpu_usage_mb: Optional[int] = None
    target_format: List[Format] = field(default_factory=lambda: TRITON_SUPPORTED_FORMATS)
    backend_accelerator: List[BackendAccelerator] = field(default_factory=lambda: [])
    tensorrt_precision: List[TensorRTPrecision] = field(default_factory=lambda: [])
    tensorrt_capture_cuda_graph: bool = False


def _load_triton_package(ctx, param, value):
    package_path = pathlib.Path(value)
    return TritonPackage(package_path)


def map_none(f, v):
    if v is not None:
        return f(v)


def _filter_models(select_config: SelectConfig, input_triton_package: TritonPackage):
    status = input_triton_package.status()
    models = []
    models_rejected = set()
    for model in status["models"]:
        if model["status"] != "succeeded":
            continue

        format_ = Format(model["conversion_config"]["target_format"])
        if format_ in [Format.TENSORRT, Format.TF_TRT, Format.TORCH_TRT]:
            model_precision = map_none(TensorRTPrecision, model["conversion_config"]["tensorrt_config"]["precision"])
        else:
            model_precision = None

        for model_store in model["model_stores"]:
            if model_store["status"] != "succeeded":
                continue
            backend_accelerator = map_none(BackendAccelerator, model_store["optimizations"].get("backend_accelerator"))
            tensorrt_precision = model_precision or map_none(
                TensorRTPrecision, model_store["optimizations"].get("tensorrt_precision")
            )
            capture_cuda_graph = model_store["optimizations"]["tensorrt_capture_cuda_graph"]

            if any(
                [
                    format_ not in select_config.target_format,
                    select_config.backend_accelerator and backend_accelerator not in select_config.backend_accelerator,
                    select_config.tensorrt_precision and tensorrt_precision not in select_config.tensorrt_precision,
                    select_config.tensorrt_capture_cuda_graph and not capture_cuda_graph,
                ]
            ):
                models_rejected.add((model_store["name"], format_))
                continue

            models.append(model_store["name"])

    if models_rejected:
        LOGGER.info(
            "Rejected following model configurations present in package due to provided constraints: %s",
            ", ".join(f"{m} ({f.value})" for m, f in models_rejected),
        )
    return models


def _filter_analyzer_results(in_file, out_file, filtered_models):
    # this is really obtuse, but at least we don't have to modify the Summary class logic
    reader = csv.DictReader(in_file)
    writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        name = row["Model"]
        if name in filtered_models:
            writer.writerow(row)


def _find_model_config(model_config_name, input_triton_package):
    with input_triton_package.open(input_triton_package.model_metrics_path) as in_file:
        reader = csv.DictReader(io.TextIOWrapper(in_file))
        for row in reader:
            config_path = row["Model Config Path"]
            if config_path == model_config_name:
                return row


def _get_top_result(select_config: SelectConfig, input_triton_package: TritonPackage):
    analysis_config = ModelAnalyzerAnalysisConfig.from_dict(
        {"objectives": select_config.objective, **dataclass2dict(select_config)}
    )
    analysis_config.top_n_configs = 1
    filtered_models = _filter_models(select_config, input_triton_package)

    with tempfile.TemporaryDirectory() as tdir:
        tpath = pathlib.Path(tdir)
        LOGGER.debug("Filtering Model Analyzer results")
        with (tpath / "metrics-model.csv").open("w") as f, input_triton_package.open(
            input_triton_package.model_metrics_path
        ) as g:
            _filter_analyzer_results(io.TextIOWrapper(g), f, filtered_models)

        with (tpath / "metrics-gpu.csv").open("w") as f, input_triton_package.open(
            input_triton_package.gpu_metrics_path
        ) as g:
            _filter_analyzer_results(io.TextIOWrapper(g), f, filtered_models)

        summary = Summary(
            results_path=tpath / "metrics-model.csv",
            metrics_path=tpath / "metrics-gpu.csv",
            analysis_config=analysis_config,
        )
        results = summary.get_results()

    LOGGER.debug("Filtering results finished")
    if not results:
        return None
    return results[0]


@click.command(name="select", help="Select models fulfilling specified constraints and produce a model store.")
@click.argument(
    "input_triton_package",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, resolve_path=True),
    required=True,
    callback=_load_triton_package,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=True, file_okay=False, exists=False, resolve_path=True),
    required=False,
    default="model_repository",
)
@click.option(
    "--override",
    help="Overwrite any existing model repository at the output path.",
    default=False,
    type=bool,
    is_flag=True,
)
@click.option(
    "-v",
    "--verbose",
    help="Provide verbose logs.",
    default=False,
    type=bool,
    is_flag=True,
)
@click.option(
    "--model-config-name",
    type=str,
    required=False,
    default=None,
    help="Pick a particular model configuration. If specified, other selection options are ignored.",
)
@cli.options_from_config(SelectConfig, SelectConfigCli)
@click.pass_context
def select_cmd(
    ctx,
    input_triton_package: TritonPackage,
    output_path: pathlib.Path,
    override: bool,
    verbose: bool,
    model_config_name: Optional[str],
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug("Running '%s'", ctx.command_path)

    select_config = SelectConfig.from_dict(kwargs)

    log_dict(
        "select args:",
        {
            **dataclass2dict(select_config),
            "override": override,
            "input_triton_package": input_triton_package,
            "output_path": output_path,
            "verbose": verbose,
            "model_config_name": model_config_name,
        },
    )

    if model_config_name is not None:
        to_extract = _find_model_config(model_config_name, input_triton_package)
    else:
        to_extract = _get_top_result(select_config, input_triton_package)

    if to_extract is None:
        sys.exit("No Model Analyzer results fulfilling given constraints found in this package.")

    input_triton_package.copy_model_to_repository(
        to_extract["Model Config Path"],
        pathlib.Path(output_path),
        output_name=to_extract["Model"],
        overwrite=override,
    )
