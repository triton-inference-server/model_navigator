#  Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import logging
from pathlib import Path
from typing import List, Tuple

from polygraphy.common import TensorMetadata

from model_navigator.optimizer.polygraphy.dataloader import DataLoader
from model_navigator.tensor import TensorSpec

LOGGER = logging.getLogger(__name__)


def get_profiling_data_path(workspace_path: Path):
    return workspace_path / "random_data.json"


def create_profiling_data(shapes: List[TensorSpec], value_ranges: List[Tuple], iterations: int, output_path: str):
    # As perf_analyzer doesn't support passing value ranges we need to generate json files
    LOGGER.debug("Generating profiling data for Perf Analyzer")
    value_ranges = dict(value_ranges)

    batch_size = 1
    input_metadata = TensorMetadata()
    for spec in shapes:
        shape = spec.shape
        batch_size = shape[0]
        input_metadata.add(spec.name, dtype=spec.dtype, shape=shape[1:])

    dataloader = DataLoader(iterations=iterations * batch_size, input_metadata=input_metadata, val_range=value_ranges)

    def _cast_input(name, value):
        min_value = value_ranges[name][0]
        target_type = type(min_value)
        return target_type(value)

    data = {
        "data": [
            {
                name: {
                    "content": [_cast_input(name, x) for x in data.flatten().tolist()],
                    "shape": list(data.shape),
                }
                for name, data in feed_dict.items()
            }
            for feed_dict in dataloader
        ]
    }

    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
