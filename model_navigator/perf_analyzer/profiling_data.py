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
import base64
import json
import logging
from pathlib import Path
from sys import getsizeof

import numpy as np

LOGGER = logging.getLogger(__name__)


def _remove_batch_dim(data):
    """Skip batch dimensions.
    This should probably be replaced by better dataset abstraction.
    """
    assert data.shape[0] == 1
    return list(data.shape[1:])


def _base64_content(data: np.ndarray):
    """Convert the array of input values to base64 format"""
    b64_content = base64.b64encode(data.flatten().tobytes()).decode("utf-8")
    return {"b64": b64_content}


def create_profiling_data(
    dataloader,
    output_path: Path,
):
    """
    Create profiling data based on dataloader and save it to JSON file.

    The perf_analyzer doesn't support passing value ranges and support the real data passed in form of JSON files. The
    real data provide better performance analysis and is mandatory for models with embeddings.

    The data is stored in form of base64 format to preserve the input data type - ex. FP16.

    More about real input data:
    https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md#real-input-data
    """
    LOGGER.debug("Generating profiling data for Perf Analyzer")

    data = {
        "data": [
            {
                name: {"content": _base64_content(data), "shape": _remove_batch_dim(data)}
                for name, data in feed_dict.items()
            }
            for idx, feed_dict in enumerate(dataloader)
        ]
    }

    LOGGER.debug(f"Saving data of size {getsizeof(data)} bytes to {output_path}")
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    LOGGER.debug("File saved")
