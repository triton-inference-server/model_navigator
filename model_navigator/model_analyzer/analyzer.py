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
from pathlib import Path

import yaml

from model_navigator.exceptions import ModelNavigatorAnalyzeException
from model_navigator.kubernetes.yaml import CustomDumper
from model_navigator.model_analyzer import AnalyzeResult, ModelAnalyzer
from model_navigator.model_analyzer.config import BaseConfigGenerator, ModelAnalyzerAnalysisConfig
from model_navigator.model_analyzer.model_analyzer import ModelAnalyzerMode
from model_navigator.model_analyzer.model_analyzer_config import ModelAnalyzerConfig
from model_navigator.model_analyzer.summary import Summary
from model_navigator.results import State, Status
from model_navigator.triton import TritonModelConfigGenerator
from model_navigator.utils import Workspace

LOGGER = logging.getLogger(__name__)


class Analyzer:
    def __init__(
        self,
        *,
        workspace: Workspace,
        model_repository: str,
        verbose: bool = False,
        analysis_config: ModelAnalyzerAnalysisConfig,
    ):
        self._workspace = workspace
        self._model_repository = workspace.path / model_repository

        self._analysis_config = analysis_config

        self._config_generator: AnalysisConfigGenerator = AnalysisConfigGenerator(
            analysis_config=self._analysis_config,
            workspace=self._workspace,
            verbose=verbose,
        )

        self._config_path = self._config_generator.analyzer_path / "config-analyze.yaml"

        self._verbose = verbose

    def run(self):
        config = self._config_generator.generate_config(self._model_repository)
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with self._config_path.open("w") as config_file:
            config_content = yaml.dump(config, Dumper=CustomDumper)
            LOGGER.debug("Model Analyzer analyze config:\n" f"{config_content}")
            config_file.write(config_content)

        quiet = self._verbose is False

        analyzer_config = ModelAnalyzerConfig()
        analyzer_config["config-file"] = self._config_path.as_posix()

        analyzer = ModelAnalyzer(config=analyzer_config)
        analyzer.run(mode=ModelAnalyzerMode.ANALYZE, verbose=self._verbose, quiet=quiet)

        LOGGER.info("Analyzer analysis done.")

        src_report = (
            self._config_generator.analyzer_path
            / "reports"
            / "summaries"
            / "Best Configs Across All Models"
            / "result_summary.pdf"
        )
        if not src_report.is_file():
            raise ModelNavigatorAnalyzeException("Model Analyzer summary report not found.")

        dst_report = self._workspace.path / "analyze_report.pdf"
        shutil.copy(src_report, dst_report)
        LOGGER.info(f"Report for best config across all models: {dst_report.resolve()}")

        return self._wrap_into_analyze_results()

    def _wrap_into_analyze_results(self):
        results = []
        summary = Summary(
            results_path=self._config_generator.results_path,
            metrics_path=self._config_generator.metrics_path,
            analysis_config=self._analysis_config,
        )

        for result in summary.get_results():
            model_path = self._model_repository / result["Model"]
            triton_config_path = (
                self._config_generator.output_model_repository_path / result["Model Config Path"] / "config.pbtxt"
            )
            triton_model_config_generator = TritonModelConfigGenerator.from_triton_config_pbtxt(
                triton_config_path, model_path
            )

            analyze_result = AnalyzeResult(
                status=Status(State.SUCCEEDED, message="Model repository analyzed successfully"),
                model_repository=self._model_repository,
                analysis_config=self._analysis_config,
                model_config_path=result["Model Config Path"],
                model_name=result["Model"],
                optimization_config=triton_model_config_generator.optimization_config,
                scheduler_config=triton_model_config_generator.scheduler_config,
                instances_config=triton_model_config_generator.instances_config,
                results_path=self._config_generator.results_path,
                metrics_path=self._config_generator.metrics_path,
            )
            results.append(analyze_result)
        return results


class AnalysisConfigGenerator(BaseConfigGenerator):
    def __init__(self, *, workspace: Workspace, analysis_config: ModelAnalyzerAnalysisConfig, verbose: int = 0):
        super().__init__(workspace=workspace, verbose=verbose)
        self._verbose = verbose
        self._analysis_config = analysis_config

        self._analyzer_results_dir_path = self._analyzer_path / "results"
        self.results_path = self._analyzer_results_dir_path / "metrics-model-inference.csv"
        self.metrics_path = self._analyzer_results_dir_path / "metrics-gpu-inference.csv"

    @property
    def results_dir_path(self):
        return self._analyzer_results_dir_path

    def generate_config(self, model_repository: Path):
        model_names = [model_dir.name for model_dir in model_repository.glob("*") if model_dir.is_dir()]
        LOGGER.info(f"Prepare analysis for {len(model_names)} models from {model_repository}:")
        for model_name in model_names:
            LOGGER.info(f"\t- {model_name}")

        # https://github.com/triton-inference-server/model_analyzer/blob/r21.07/docs/config.md
        config = {
            "analysis_models": model_names,
            "checkpoint_directory": self._analyzer_checkpoints_dir_path.as_posix(),
            "output_model_repository_path": self.output_model_repository_path.as_posix(),
            "export_path": self._analyzer_path.resolve().as_posix(),
            "filename_model_inference": self.results_path.name,
            "filename_model_gpu": self.metrics_path.name,
            "num_top_model_configs": self._analysis_config.top_n_configs,
            "objectives": self._analysis_config.objectives,
        }

        constraints = self._get_constraints()
        if constraints:
            config["constraints"] = constraints

        return config

    def _get_constraints(self):
        constraints = {}
        if self._analysis_config.max_latency_ms is not None:
            constraints["perf_latency"] = {"max": self._analysis_config.max_latency_ms}

        if self._analysis_config.min_throughput is not None:
            constraints["perf_throughput"] = {"min": self._analysis_config.min_throughput}

        if self._analysis_config.max_gpu_usage_mb is not None:
            constraints["gpu_used_memory"] = {"max": self._analysis_config.max_gpu_usage_mb}

        return constraints
