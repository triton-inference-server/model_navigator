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
import heapq
import json
import logging
from pathlib import Path
from typing import Dict, List

import tabulate

from model_navigator.model_analyzer.config import ModelAnalyzerAnalysisConfig
from model_navigator.model_analyzer.measurement import Measurement
from model_navigator.model_analyzer.result_comparator import ResultComparator

LOGGER = logging.getLogger(__name__)


class Summary:
    def __init__(
        self,
        *,
        results_path: Path,
        metrics_path: Path,
        analysis_config: ModelAnalyzerAnalysisConfig,
    ):
        self._results_path = results_path
        self._metrics_path = metrics_path
        self._analysis_config = analysis_config

        self._inference = []
        self._gpu_metrics = []

    def show(self):
        self._prepare()
        if len(self._inference) == 0:
            LOGGER.warning("No matching models has been found for given constraints.")
            return

        LOGGER.info("Selected models based on passed constraints and objectives:")
        self._show_results(self._inference, section="Models (Inference)")
        self._show_results(self._gpu_metrics, section="Models (GPU Metrics)")

    def get_results(self):
        self._prepare()
        return self._inference

    def get_metrics(self):
        self._prepare()
        return self._gpu_metrics

    def get_result(self, idx):
        self._prepare()
        return self._inference[idx]

    def get_metric(self, idx):
        self._prepare()
        return self._gpu_metrics[idx]

    def _show_results(self, results: List[Dict], section: str):
        print(f"{section}:")
        if results:
            max_index = min(len(results), self._analysis_config.top_n_configs)
            index = range(1, max_index + 1)
            header = ["No"] + list(results[0].keys())
            summary = map(lambda x: list(map(lambda item: item[1], x.items())), results)

            print(tabulate.tabulate(summary, headers=header, showindex=index, tablefmt="plain"))
        else:
            print("<Missing data>")

    def _prepare(self):
        if len(self._inference) > 0:
            return

        results = self._rows_from_csv(file_path=self._results_path)
        metrics = self._rows_from_csv(file_path=self._metrics_path)

        inference = self._filter(results)
        inference = self._top_results(inference)
        metrics = self._top_metrics(metrics, inference)

        self._inference = inference
        self._gpu_metrics = metrics

    def _rows_from_csv(self, *, file_path: Path) -> List[Dict]:
        data = []
        if not file_path.exists():
            return data

        with file_path.open("r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)

        return data

    def _filter(self, results: List[Dict]):
        results = list(filter(lambda item: item["Satisfies Constraints"].lower() == "yes", results))
        return results

    def _top_results(self, results: List[Dict]):
        comparator = ResultComparator(metric_objectives=self._analysis_config.objectives)
        measurements = []
        for result in results:
            heapq.heappush(measurements, Measurement(result, comparator))

        n = self._analysis_config.top_n_configs
        top_measurements = heapq.nsmallest(min(n, len(measurements)), measurements)
        top_results = [measurement.result for measurement in top_measurements]

        return top_results

    def _top_metrics(self, metrics: List[Dict], results: List[Dict]):
        metrics_hashed = {self._hash_result(metric): metric for metric in metrics}
        results_hashes = [self._hash_result(result) for result in results]
        filtered_metrics = [
            metrics_hashed[result_hash] for result_hash in results_hashes if result_hash in metrics_hashed
        ]
        return filtered_metrics

    def _hash_result(self, result: Dict):
        merge_fields = {
            "Model",
            "Batch",
            "Concurrency",
            "Model Config Path",
            "Instance Group",
            "Dynamic Batching Sizes",
            "Satisfies Constraints",
        }

        filtered_items = {}
        for key, value in result.items():
            if key in merge_fields:
                filtered_items[key] = value

        unique_key = hash(json.dumps(filtered_items, sort_keys=True))

        return unique_key
