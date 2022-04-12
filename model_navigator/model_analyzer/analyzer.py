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
from model_navigator.triton import TritonModelConfigGenerator, TritonModelStore
from model_navigator.utils import Workspace

LOGGER = logging.getLogger(__name__)


class Analyzer:
    def __init__(
        self,
        *,
        workspace: Workspace,
        model_repository: Path,
        verbose: bool = False,
        analysis_config: ModelAnalyzerAnalysisConfig,
    ):
        self._workspace = workspace
        self._model_repository = model_repository

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

        report_dir = self._config_generator.analyzer_path / "reports" / "summaries" / "Best Configs Across All Models"
        expected_report_paths = [report_dir / "result_summary.pdf", report_dir / "result_summary.html"]
        existing_report_paths = [report_path for report_path in expected_report_paths if report_path.is_file()]
        if not existing_report_paths:
            raise ModelNavigatorAnalyzeException(
                f"Model Analyzer summary report not found. "
                f"Expected: {', '.join([report_path.as_posix() for report_path in expected_report_paths])}"
            )
        src_report_path = existing_report_paths[0]
        report_suffix = "".join(src_report_path.suffixes)
        dst_report_path = self._workspace.path / f"analyze_report{report_suffix}"
        shutil.copy(src_report_path, dst_report_path)
        LOGGER.info(f"Report for best config across all models: {dst_report_path.resolve()}")

        return self._wrap_into_analyze_results()

    def _wrap_into_analyze_results(self):
        results = []
        summary = Summary(
            results_path=self._config_generator.results_path,
            metrics_path=self._config_generator.metrics_path,
            analysis_config=self._analysis_config,
        )

        for result in summary.get_results():
            model_store = TritonModelStore(self._model_repository)
            external_model_path = model_store.get_model_path(result["Model"])

            triton_config_path = (
                self._config_generator.output_model_repository_path / result["Model Config Path"] / "config.pbtxt"
            )
            triton_model_config_generator = TritonModelConfigGenerator.parse_triton_config_pbtxt(
                triton_config_path, external_model_path=external_model_path
            )

            analyze_result = AnalyzeResult(
                status=Status(State.SUCCEEDED, message="Model repository analyzed successfully"),
                model_repository=self._model_repository,
                analysis_config=self._analysis_config,
                model_config_path=result["Model Config Path"],
                model_name=result["Model"],
                optimization_config=triton_model_config_generator.optimization_config,
                batching_config=triton_model_config_generator.batching_config,
                dynamic_batching_config=triton_model_config_generator.dynamic_batching_config,
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

        # https://github.com/triton-inference-server/model_analyzer/blob/r22.02/docs/config.md
        inference_output_fields = [
            "model_name",
            "batch_size",
            "concurrency",
            "model_config_path",
            "instance_group",
            "satisfies_constraints",
            "perf_throughput",
            "perf_latency_avg",
            "perf_latency_p90",
            "perf_latency_p95",
            "perf_latency_p99",
            "perf_client_response_wait",
            "perf_client_send_recv",
            "perf_server_queue",
            "perf_server_compute_input",
            "perf_server_compute_infer",
            "perf_server_compute_output",
        ]

        config = {
            "analysis_models": model_names,
            "checkpoint_directory": self._analyzer_checkpoints_dir_path.as_posix(),
            "output_model_repository_path": self.output_model_repository_path.as_posix(),
            "export_path": self._analyzer_path.resolve().as_posix(),
            "filename_model_inference": self.results_path.name,
            "filename_model_gpu": self.metrics_path.name,
            "num_top_model_configs": self._analysis_config.top_n_configs,
            "objectives": self._analysis_config.objectives,
            "inference_output_fields": inference_output_fields,
        }

        constraints = self._get_constraints()
        if constraints:
            config["constraints"] = constraints

        return config

    def _get_constraints(self):
        constraints = {}
        if self._analysis_config.max_latency_ms is not None:
            constraints["perf_latency_p99"] = {"max": self._analysis_config.max_latency_ms}

        if self._analysis_config.min_throughput is not None:
            constraints["perf_throughput"] = {"min": self._analysis_config.min_throughput}

        if self._analysis_config.max_gpu_usage_mb is not None:
            constraints["gpu_used_memory"] = {"max": self._analysis_config.max_gpu_usage_mb}

        return constraints
