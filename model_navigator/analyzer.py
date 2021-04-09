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
import os
import shutil
from pathlib import Path

import yaml

from .catalog import Catalog
from .config import ModelNavigatorBaseConfig
from .model import Model
from .model_analyzer import ModelAnalyzer, ModelAnalyzerConfig
from .model_navigator_exceptions import ModelNavigatorException
from .optimizer.transformers import get_optimized_models_dir
from .perf_analyzer import PerfAnalyzerConfig
from .perf_analyzer.profiling_data import get_profiling_data_path

LOGGER = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, config: ModelNavigatorBaseConfig):
        self._config = config
        self._catalog = Catalog()
        workspace_path = Path(config.workspace_path)
        self._analyzer_path = workspace_path / "analyzer"
        self._model_repository = self._analyzer_path / "model-store"
        self._model_analyzer_repository = self._analyzer_path / "output-model-store"
        self._analyzer_results = self._analyzer_path / "results"
        self._analyzer_config = self._analyzer_path / "config.yaml"
        self._analyzer_triton_log = self._analyzer_path / "triton.log"
        self._profiling_data_path = get_profiling_data_path(workspace_path)
        self._optimizer_path = get_optimized_models_dir(workspace_path)

        self._filename_results_inference = "metrics-model-inference.csv"
        self._filename_metrics_inference = "metrics-gpu-inference.csv"

        self._prepare_catalogs()

    def add(self, model: Model):
        self._move_to_model_store(model)
        self._catalog.add(model)

    def results_file(self) -> Path:
        return self._analyzer_results / self._filename_results_inference

    def metrics_file(self) -> Path:
        return self._analyzer_results / self._filename_metrics_inference

    def run(self):
        if len(self._catalog) == 0:
            raise ModelNavigatorException("No models found for analysis. Please, review error logs.")

        LOGGER.info(f"Prepare analysis for {len(self._catalog)} models:")
        for model in self._catalog:
            LOGGER.info(model.name)

        self._prepare_analyzer_config()

        analyzer_params = {
            "config-file": self._analyzer_config.as_posix(),
            "log-level": "DEBUG" if self._config.verbose else "INFO",
        }

        analyzer_config = ModelAnalyzerConfig()
        for key, value in analyzer_params.items():
            analyzer_config[key] = value

        analyzer = ModelAnalyzer(config=analyzer_config)
        try:
            analyzer.run()
        except ModelNavigatorException as e:
            raise e

        LOGGER.info("Analysis done.")

    def _move_to_model_store(self, model):
        dir_name = model.triton_path.name
        src_dir = model.triton_path
        dst_dir = self._model_repository / dir_name
        shutil.move(src_dir.as_posix(), dst_dir.as_posix())
        LOGGER.info(f"Copying files from {src_dir} to {dst_dir}")
        model.triton_path = dst_dir

    def _prepare_analyzer_config(self):
        model_configuration = self._model_configuration
        if len(model_configuration) > 0:
            model_names = dict()
            for model in self._catalog:
                model_names[model.name] = dict(model_configuration)
        else:
            model_names = list()
            for model in self._catalog:
                model_names.append(model.name)

        config = {
            "triton_docker_image": f"nvcr.io/nvidia/tritonserver:{self._config.triton_version}",
            "triton_launch_mode": self._config.triton_launch_mode,
            "model_repository": self._model_repository.as_posix(),
            "output_model_repository_path": self._model_analyzer_repository.as_posix(),
            "export": True,
            "export_path": self._analyzer_path.as_posix(),
            "filename_model_inference": self._filename_results_inference,
            "filename_model_gpu": self._filename_metrics_inference,
            "perf_analyzer_cpu_util": 400,  # WAR - to avoid killing perf analyzer
            "perf_measurement_window": 10000,  # WAR - to have valid measurement window always
            "triton_server_flags": {"strict-model-config": False},
            "run_config_search_max_concurrency": self._config.max_concurrency,
            "run_config_search_max_instance_count": self._config.max_instance_count,
            "run_config_search_max_preferred_batch_size": self._config.max_preferred_batch_size,
            "model_names": model_names,
            "top_n_configs": self._config.top_n_configs,
            "objectives": self._config.objectives,
            "summarize": False,
        }

        if self._config.verbose:
            config["perf_output"] = True
            config["triton_output_path"] = self._analyzer_triton_log.as_posix()

        content = yaml.dump(config)
        LOGGER.debug("Model Analyzer config:\n" f"{content}")

        if self._analyzer_config.is_file():
            os.remove(self._analyzer_config.as_posix())

        with open(self._analyzer_config, "w") as f:
            f.write(content)

    @property
    def _constraints(self):
        constraints = dict()
        if self._config.max_latency_ms:
            constraints["perf_latency"] = {"max": self._config.max_latency_ms}

        if self._config.min_throughput:
            constraints["perf_throughput"] = {"min": self._config.min_throughput}

        if self._config.max_gpu_usage_mb:
            constraints["gpu_used_memory"] = {"max": self._config.max_gpu_usage_mb}

        return constraints

    @property
    def _model_configuration(self):
        configuration = dict()
        if self._config.concurrency:
            configuration["parameters"] = {"concurrency": self._config.concurrency}

        model_config = dict()
        if self._config.instance_counts:
            items = [{"kind": "KIND_GPU", "count": self._config.instance_counts}]
            model_config["instance_group"] = items

        if self._config.preferred_batch_sizes:
            preferred_batch_sizes = []
            for batch_sizes in self._config.preferred_batch_sizes:
                preferred_batch_sizes.append(batch_sizes)

            model_config["dynamic_batching"] = {"preferred_batch_size": preferred_batch_sizes}

        constraints = self._constraints
        if constraints:
            configuration["constraints"] = constraints

        if model_config:
            configuration["model_config_parameters"] = model_config

        # TODO: what if we provide shapes but model have no dynamic axes?
        if self._config.value_ranges:
            configuration["perf_analyzer_flags"] = {"input-data": self._profiling_data_path.as_posix()}
        elif self._config.max_shapes:
            configuration["perf_analyzer_flags"] = {
                "shape": " ".join(
                    [PerfAnalyzerConfig.shape_param_from_tensor_spec(spec) for spec in self._config.max_shapes]
                )
            }

        return configuration

    def _prepare_catalogs(self):
        self._prepare_analyzer_catalog()
        self._prepare_model_repository()
        self._prepare_results_catalog()

    def _prepare_model_repository(self):
        if self._model_repository.is_dir():
            shutil.rmtree(self._model_repository.as_posix())

        self._model_repository.mkdir(parents=True)

        if self._model_analyzer_repository.is_dir():
            shutil.rmtree(self._model_analyzer_repository.as_posix())

        self._model_analyzer_repository.mkdir(parents=True)

    def _prepare_analyzer_catalog(self):
        if self._analyzer_path.is_dir():
            shutil.rmtree(self._analyzer_path.as_posix())

        self._analyzer_path.mkdir(parents=True)
        self._analyzer_results.mkdir(parents=True)

    def _prepare_results_catalog(self):
        if self._analyzer_results.is_dir():
            shutil.rmtree(self._analyzer_results.as_posix())

        self._analyzer_results.mkdir(parents=True)
