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

from .analyzer import Analyzer
from .catalog import Catalog
from .chart import ChartGenerator
from .config.config import ModelNavigatorBaseConfig
from .configurator import Configurator
from .deployer import Deployer
from .log import FileLogger
from .model import InputModel
from .model_navigator_exceptions import ModelNavigatorDeployerException
from .optimizer.pipelines import BaseModelPipeline
from .perf_analyzer.profiling_data import create_profiling_data, get_profiling_data_path
from .result import Summary

LOGGER = logging.getLogger(__name__)


class ModelNavigator:
    def __init__(self, config: ModelNavigatorBaseConfig):
        self._config = config
        self._configurator = Configurator()
        self._deployer = Deployer(config=self._config)
        self._analyzer = Analyzer(config=self._config)
        self._chart_generator = ChartGenerator(config=self._config)
        self._catalog = Catalog()
        self._summary = None

    def run(self):
        input_model = InputModel(
            name=self._config.model_name,
            path=Path(self._config.model_path),
            config=self._config,
        )

        input_models = Catalog()
        input_models.add(input_model)

        pipeline = BaseModelPipeline.for_model(model=input_model, config=self._config, run_in_container=True)
        for optimized_model in pipeline.execute(input_model, config=self._config):
            LOGGER.info(f"New model: {optimized_model.name} @ {optimized_model.path}")
            LOGGER.debug(f"    config: {optimized_model.config}")
            input_models.add(optimized_model)

        value_ranges = self._config.value_ranges
        if value_ranges:
            profiling_data_path = get_profiling_data_path(Path(self._config.workspace_path))
            self._generate_profiling_data(
                shapes=self._config.max_shapes, value_ranges=value_ranges, output_path=profiling_data_path
            )

        LOGGER.info(f"Number of models after optimization: {len(input_models)}")
        if len(self._config.preferred_batch_sizes) > 0:
            max_batch_size = 0
            for batch_sizes in self._config.preferred_batch_sizes:
                max_batch_size = max(max(batch_sizes), max_batch_size)
        else:
            max_batch_size = self._config.max_preferred_batch_size

        self._catalog = self._configurator.get_models_variants(models=input_models, max_batch_size=max_batch_size)

        self._init_log_directory()

        LOGGER.info(f"Prepared {len(self._catalog)} model variants.")
        for idx, model in enumerate(self._catalog, start=1):
            LOGGER.info(f"Verifying variant {idx} ")
            try:
                self._deployer.deploy_model(model)
            except ModelNavigatorDeployerException:
                LOGGER.warning(
                    f"Deployment for {model.name} variant failed."
                    f"In order to investigate problem see error log saved in: {model.error_log}"
                )
                continue

            LOGGER.info(f"Deployment for {model.name} variant succeed. Promoting to analysis stage.")
            self._analyzer.add(model)

        self._report_failing_models()

        self._analyzer.run()

        self._prepare_summary()
        self._show_summary()
        self._prepare_charts(base_model=input_model)

    def _generate_profiling_data(self, *, shapes, value_ranges, output_path):
        # As perf_analyzer doesn't support passing value ranges we need to generate json files
        inputs_with_missing_shapes = sorted(set(dict(value_ranges)) - set([spec.name for spec in shapes]))
        if inputs_with_missing_shapes:
            raise ValueError(
                f"There are missing shapes for {', '.join(inputs_with_missing_shapes)}. "
                f"Use max-shapes to define missing shapes."
            )
        create_profiling_data(shapes, value_ranges=value_ranges, iterations=128, output_path=output_path.as_posix())

    def _prepare_summary(self):
        self._summary = Summary(
            results_file=self._analyzer.results_file(),
            metrics_file=self._analyzer.metrics_file(),
            config=self._config,
        )

    def _show_summary(self):
        self._summary.show()

    def _prepare_charts(self, base_model: InputModel):
        self._chart_generator.run(
            base_model=base_model,
            summary=self._summary,
            catalog=self._catalog,
        )

    def _init_log_directory(self):
        logs_dir = FileLogger.get_logs_dir(self._config)
        if logs_dir.is_dir():
            shutil.rmtree(logs_dir.as_posix())

        logs_dir.mkdir(parents=True)

    def _report_failing_models(self):
        failing_models = list()
        for model in filter(lambda m: m.error_log is not None, self._catalog):
            failing_models.append(model)

        if len(failing_models) > 0:
            LOGGER.warning("Unable to deploy some model variants. Please review logs to learn more about issues:")
            for model in failing_models:
                LOGGER.warning(model.error_log)
