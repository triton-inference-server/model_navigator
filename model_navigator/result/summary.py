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
from typing import Dict, List, Union

import copy
import csv
import heapq
import logging
from pathlib import Path

from tabulate import tabulate

from ..config import ModelNavigatorBaseConfig
from ..result.measurement import Measurement
from ..result.result_comparator import ResultComparator

LOGGER = logging.getLogger(__name__)


class Summary:
    def __init__(
        self, results_file: Union[str, Path], metrics_file: Union[str, Path], config: ModelNavigatorBaseConfig
    ):
        self._results_file = results_file
        self._metrics_file = metrics_file
        self._config = config

        self._results = list()

    def show(self):
        self._prepare()
        if len(self._results) == 0:
            LOGGER.warning("No matching models has been found for given constraints.")
            return

        headers = list(self._results[0].keys())
        summary = map(lambda x: list(map(lambda item: item[1], x.items())), self._results)
        LOGGER.info("Selected models based on passed constraints and objectives:")
        print(tabulate(summary, headers=headers, showindex=range(1, len(self._results) + 1)))

    def get(self, idx):
        self._prepare()
        return self._results[idx]

    def _prepare(self):
        if len(self._results) > 0:
            return

        results = self._rows_from_csv(file=self._results_file)
        metrics = self._rows_from_csv(file=self._metrics_file)

        results = self._merge(results, metrics)
        results = self._filter(results)
        results = self._top_results(results)

        self._results = results

    def _rows_from_csv(self, *, file: Union[str, Path]) -> List[Dict]:
        data = list()
        with open(file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)

        return data

    def _merge(self, results: List[Dict], metrics: List[Dict]):
        merge_fields = {
            "Model",
            "Batch",
            "Concurrency",
            "Model Config Path",
            "Instance Group",
            "Dynamic Batcher Sizes",
            "Satisfies Constraints",
        }

        metrics_cpy = copy.deepcopy(metrics)

        def _matching_metrics(result: Dict, metrics: List[Dict]):
            for idx, metric in enumerate(metrics):
                match = True
                for field in merge_fields:
                    if metric[field] != result[field]:
                        match = False
                        break
                if not match:
                    continue

                return idx, metric

            raise ValueError(f"No matching metric for result. {result} Please, verify analyzer results.")

        merged_results = list()
        for result in results:
            idx, metric = _matching_metrics(result, metrics_cpy)
            metrics_cpy.pop(idx)

            metric = {key: value for key, value in metric.items() if key not in merge_fields}

            merged_result = {**result, **metric}
            merged_results.append(merged_result)

        return merged_results

    def _filter(self, results: List[Dict]):
        results = list(filter(lambda item: item["Satisfies Constraints"].lower() == "yes", results))
        return results

    def _top_results(self, results: List[Dict]):
        comparator = ResultComparator(metric_objectives=self._config.objectives)
        measurements = list()
        for result in results:
            heapq.heappush(measurements, Measurement(result, comparator))

        n = self._config.top_n_configs
        top_measurements = heapq.nsmallest(min(n, len(measurements)), measurements)
        top_results = [measurement.result for measurement in top_measurements]

        return top_results
