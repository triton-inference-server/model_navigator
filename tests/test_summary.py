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
from argparse import Namespace
from pathlib import Path

from model_navigator.config import ModelNavigatorConfig
from model_navigator.record.types.perf_latency import PerfLatency
from model_navigator.record.types.perf_throughput import PerfThroughput
from model_navigator.result import Summary


def _init_config():
    args = Namespace(model_name="model", model_path="path")
    config = ModelNavigatorConfig()
    config.set_config_values(args=args)

    return config


def test_merge_results():
    config = _init_config()
    file_dir = Path(__file__).parent.absolute()
    results_file = file_dir / "files" / "results.csv"
    metrics_file = file_dir / "files" / "metrics.csv"

    summary = Summary(results_file=results_file, metrics_file=metrics_file, config=config)

    results = summary._rows_from_csv(file=results_file)
    metrics = summary._rows_from_csv(file=metrics_file)

    merged_results = summary._merge(results, metrics)

    assert len(results) == len(merged_results)


def test_filter_results():
    config = _init_config()

    file_dir = Path(__file__).parent.absolute()
    results_file = file_dir / "files" / "results.csv"
    metrics_file = file_dir / "files" / "metrics.csv"

    summary = Summary(results_file=results_file, metrics_file=metrics_file, config=config)

    results = summary._rows_from_csv(file=results_file)
    metrics = summary._rows_from_csv(file=metrics_file)

    merged_results = summary._merge(results, metrics)
    filtered_results = summary._filter(merged_results)

    assert len(filtered_results) == 10


def test_top_results_perf():
    config = _init_config()

    file_dir = Path(__file__).parent.absolute()
    results_file = file_dir / "files" / "results.csv"
    metrics_file = file_dir / "files" / "metrics.csv"

    summary = Summary(results_file=results_file, metrics_file=metrics_file, config=config)

    results = summary._rows_from_csv(file=results_file)
    metrics = summary._rows_from_csv(file=metrics_file)

    merged_results = summary._merge(results, metrics)
    filtered_results = summary._filter(merged_results)
    top_results = summary._top_results(filtered_results)

    assert config.top_n_configs == 3
    assert len(top_results) == 3
    for idx in range(1, len(top_results)):
        assert top_results[idx - 1][PerfThroughput.header()] >= top_results[idx][PerfThroughput.header()]


def test_top_results_latency():
    config = _init_config()
    config.objectives = {"perf_latency": 10}

    file_dir = Path(__file__).parent.absolute()
    results_file = file_dir / "files" / "results.csv"
    metrics_file = file_dir / "files" / "metrics.csv"

    summary = Summary(results_file=results_file, metrics_file=metrics_file, config=config)

    results = summary._rows_from_csv(file=results_file)
    metrics = summary._rows_from_csv(file=metrics_file)

    merged_results = summary._merge(results, metrics)
    filtered_results = summary._filter(merged_results)
    top_results = summary._top_results(filtered_results)

    assert config.top_n_configs == 3
    assert len(top_results) == 3
    for idx in range(1, len(top_results)):
        assert top_results[idx - 1][PerfLatency.header()] <= top_results[idx][PerfLatency.header()]


def test_top_results_wighted():
    config = _init_config()
    config.objectives = {"perf_throughput": 10, "perf_latency": 5}

    file_dir = Path(__file__).parent.absolute()
    results_file = file_dir / "files" / "results.csv"
    metrics_file = file_dir / "files" / "metrics.csv"

    summary = Summary(results_file=results_file, metrics_file=metrics_file, config=config)

    results = summary._rows_from_csv(file=results_file)
    metrics = summary._rows_from_csv(file=metrics_file)

    merged_results = summary._merge(results, metrics)
    filtered_results = summary._filter(merged_results)
    top_results = summary._top_results(filtered_results)

    assert config.top_n_configs == 3
    assert len(top_results) == 3
    # 1st choice
    assert top_results[0][PerfLatency.header()] <= top_results[1][PerfLatency.header()]
    assert top_results[0][PerfThroughput.header()] <= top_results[1][PerfThroughput.header()]

    # 2nd choice
    assert top_results[1][PerfLatency.header()] <= top_results[2][PerfLatency.header()]
    assert top_results[1][PerfThroughput.header()] >= top_results[2][PerfThroughput.header()]


def test_prepare_results():
    config = _init_config()

    file_dir = Path(__file__).parent.absolute()
    results_file = file_dir / "files" / "results.csv"
    metrics_file = file_dir / "files" / "metrics.csv"

    summary = Summary(results_file=results_file, metrics_file=metrics_file, config=config)

    summary._prepare()

    assert config.top_n_configs == 3
    assert len(summary._results) == config.top_n_configs
