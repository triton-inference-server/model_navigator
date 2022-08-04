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

import click

from model_navigator.cli.optimize import optimize_cmd
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
    TritonBatchingConfigCli,
    TritonCustomBackendParametersConfigCli,
    TritonModelInstancesConfigCli,
)
from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.converter import ComparatorConfig, ConversionLaunchMode, ConversionSetConfig, DatasetProfileConfig
from model_navigator.model import ModelConfig, ModelSignatureConfig
from model_navigator.model_analyzer import ModelAnalyzerAnalysisConfig, ModelAnalyzerProfileConfig
from model_navigator.perf_analyzer import PerfMeasurementConfig
from model_navigator.triton import TritonBatchingConfig, TritonModelInstancesConfig
from model_navigator.triton.config import RunTritonConfig, TritonCustomBackendParametersConfig
from model_navigator.utils import cli

LOGGER = logging.getLogger("run")


@click.command(name="run", help="Alias for optimize", deprecated=True)
@cli.common_options
@cli.options_from_config(ModelConfig, ModelConfigCli)
@cli.options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
@cli.options_from_config(TritonBatchingConfig, TritonBatchingConfigCli)
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
def run_cmd(
    ctx,
    *args,
    **kwargs,
):
    return ctx.forward(optimize_cmd, *args, **kwargs)
