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
from typing import List

import click

from model_navigator.cli.spec import ModelAnalyzerAnalysisConfigCli
from model_navigator.cli.utils import exit_cli_command, is_cli_command
from model_navigator.log import init_logger, log_dict
from model_navigator.model_analyzer.analyzer import Analyzer
from model_navigator.model_analyzer.config import ModelAnalyzerAnalysisConfig
from model_navigator.model_analyzer.results import AnalyzeResult
from model_navigator.model_analyzer.summary import Summary
from model_navigator.results import ResultsStore, State, Status
from model_navigator.utils import Workspace
from model_navigator.utils.cli import common_options, options_from_config
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("analyze")


@click.command(name="analyze", help="Analyze models using Triton Model Analyzer")
@common_options
@click.option(
    "--model-repository",
    type=click.Path(file_okay=False),
    required=True,
    help="Path to the Triton Model Repository.",
)
@options_from_config(ModelAnalyzerAnalysisConfig, ModelAnalyzerAnalysisConfigCli)
@click.pass_context
def analyze_cmd(
    ctx,
    verbose: bool,
    workspace_path: str,
    model_repository: str,
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {kwargs.get('config_path')}")

    run_command_validators(
        ctx.command.name,
        configuration={
            "verbose": verbose,
            "workspace_path": workspace_path,
            "model_repository": model_repository,
            **kwargs,
        },
    )

    workspace = Workspace(workspace_path)

    analysis_config = ModelAnalyzerAnalysisConfig.from_dict(kwargs)

    if verbose:
        log_dict(
            "analyze args:",
            {
                **dataclasses.asdict(analysis_config),
                "workspace_path": workspace.path,
                "verbose": verbose,
            },
        )

    analyzer = Analyzer(
        workspace=workspace,
        model_repository=Path(model_repository),
        verbose=verbose,
        analysis_config=analysis_config,
    )

    try:
        analyze_results = analyzer.run()
        if analyze_results:
            _prepare_summary(analysis_config=analysis_config, analyze_results=analyze_results)

    except Exception:
        message = traceback.format_exc()
        LOGGER.warning(f"Encountered exception \n{message}")
        analyze_results = [
            AnalyzeResult(
                status=Status(State.FAILED, message=message),
                model_repository=workspace.path / model_repository,
                analysis_config=analysis_config,
                results_path=None,
                metrics_path=None,
            )
        ]

    results_store = ResultsStore(workspace)
    results_store.dump(ctx.command.name.replace("-", "_"), analyze_results)

    failed_analyze_results = [result for result in analyze_results if result.status.state == State.FAILED]
    if failed_analyze_results and is_cli_command(ctx):
        exit_cli_command(failed_analyze_results[0].status)

    return analyze_results


def _prepare_summary(
    *,
    analysis_config: ModelAnalyzerAnalysisConfig,
    analyze_results: List[AnalyzeResult],
):
    summary = Summary(
        results_path=analyze_results[0].results_path,
        metrics_path=analyze_results[0].metrics_path,
        analysis_config=analysis_config,
    )
    summary.show()
