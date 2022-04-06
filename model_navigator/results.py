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
import dataclasses
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import yaml

from model_navigator.exceptions import ModelNavigatorException
from model_navigator.utils import Workspace
from model_navigator.utils.config import dataclass2dict, dict2dataclass

LOGGER = logging.getLogger(__name__)


class State(Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclasses.dataclass
class Status:
    state: State
    message: str
    log_path: Optional[str] = None


class ResultsStore:
    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def dump(self, stage, results: Sequence) -> Path:
        results_path: Path = self.get_path(stage)
        LOGGER.debug(f"Saving results of {stage} stage into {results_path}")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as results_file:
            for result in results:
                LOGGER.debug(result)
            results_dict = [dataclass2dict(result) for result in results]
            yaml.dump(results_dict, results_file)
        return results_path

    def load(self, stage):
        # TODO: move this to decorator which is called in cli.main() by iterating through commands
        # TODO: missing results for run command
        from model_navigator.cli.triton_config_model import ConfigModelResult
        from model_navigator.configurator import TritonConfiguratorResult
        from model_navigator.converter import ConversionResult
        from model_navigator.kubernetes import HelmChartGenerationResult
        from model_navigator.model_analyzer import AnalyzeResult, ProfileResult

        _CLS_FOR_STAGE = {
            "convert_model": ConversionResult,
            "triton_config_model": ConfigModelResult,
            "configure_models_on_triton": TritonConfiguratorResult,
            "profile": ProfileResult,
            "analyze": AnalyzeResult,
            "helm_chart_create": HelmChartGenerationResult,
        }

        cls = _CLS_FOR_STAGE[stage]
        results_path: Path = self.get_path(stage)
        if not results_path.exists():
            raise ModelNavigatorException(f"No results found for {stage}")

        with results_path.open("r") as results_file:
            results = yaml.safe_load(results_file)
            results = [dict2dataclass(cls, result) for result in results]
        return results

    def get_path(self, stage: str) -> Path:
        return self._workspace.path / f"{stage}_results.yaml"
