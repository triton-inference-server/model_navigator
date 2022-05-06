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
import json
import logging
from pathlib import Path
from sys import getsizeof

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_DATA_FILENAME = "random_data.json"


def get_profiling_data_path(workspace_path: Path):
    return workspace_path / DEFAULT_RANDOM_DATA_FILENAME


def _remove_batch_dim(data):
    """Skip batch dimensions.
    This should probably be replaced by better dataset abstraction.
    """
    assert data.shape[0] == 1
    return list(data.shape[1:])


def create_profiling_data(
    dataloader,
    output_path: Path,
):
    # As perf_analyzer doesn't support passing value ranges we need to generate json files
    LOGGER.debug("Generating profiling data for Perf Analyzer")

    data = {
        "data": [
            {
                name: {"content": data.flatten().tolist(), "shape": _remove_batch_dim(data)}
                for name, data in feed_dict.items()
            }
            for idx, feed_dict in enumerate(dataloader)
        ]
    }

    LOGGER.debug(f"Saving data of size {getsizeof(data)} bytes to {output_path}")
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    LOGGER.debug("File saved")
