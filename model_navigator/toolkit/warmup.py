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
from typing import List, Optional

import logging
import sys

from model_navigator.model_navigator_exceptions import ModelNavigatorException
from model_navigator.perf_analyzer import PerfAnalyzer, PerfAnalyzerConfig
from model_navigator.triton.client import parse_server_url

LOGGER = logging.getLogger(__name__)


def warmup(
    model_name: str,
    batch_sizes: List[int],
    number_of_model_instances: int = 1,
    triton_instances: int = 1,
    profiling_data: str = "random",
    input_shapes: Optional[List[str]] = None,
    server_url: str = "http://localhost:8000",
    measurement_window: int = 10000,
    shared_memory: bool = False,
):
    LOGGER.info("====== Warmup start ======")

    protocol, host, port = parse_server_url(server_url)
    input_shapes = " ".join(map(lambda shape: f" --shape {shape}", input_shapes)) if input_shapes else ""

    measurement_window = 6 * measurement_window

    max_batch_size = max(batch_sizes)
    max_total_requests = 2 * max_batch_size * triton_instances * number_of_model_instances
    max_concurrency = min(256, max_total_requests)
    batch_size = max(1, max_total_requests // 256)

    step = max(1, max_concurrency // 2)
    min_concurrency = step

    params = {
        "model-name": model_name,
        "model-version": 1,
        "batch-size": batch_size,
        "url": f"{host}:{port}",
        "protocol": protocol,
        "concurrency-range": f"{min_concurrency}:{max_concurrency}:{step}",
        "input-data": profiling_data,
        "measurement-interval": measurement_window,
        "verbose": True,
    }

    if input_shapes:
        params["shapes"] = input_shapes

    if shared_memory:
        params["shared-memory"] = "cuda"

    perf_config = PerfAnalyzerConfig()
    for param, value in params.items():
        perf_config[param] = value

    perf_analyzer = PerfAnalyzer(perf_config, stream_output=True)
    try:
        perf_analyzer.run()
    except ModelNavigatorException as e:
        LOGGER.error(e.message())
        sys.exit(1)

    LOGGER.info("====== Warmup done ======")
