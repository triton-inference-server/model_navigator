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
from sys import getsizeof
from typing import Dict, Tuple

import numpy as np
from polygraphy.common import TensorMetadata

from model_navigator.converter import DataLoader

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_DATA_FILENAME = "random_data.json"


def get_profiling_data_path(workspace_path: Path):
    return workspace_path / DEFAULT_RANDOM_DATA_FILENAME


def create_profiling_data(
    shapes: Dict[str, Tuple],
    value_ranges: Dict[str, Tuple],
    dtypes: Dict[str, np.dtype],
    iterations: int,
    output_path: Path,
):
    # As perf_analyzer doesn't support passing value ranges we need to generate json files
    LOGGER.debug("Generating profiling data for Perf Analyzer")

    batch_size = 1
    input_metadata = TensorMetadata()
    for name, shape in shapes.items():
        batch_size = shape[0]
        dtype = dtypes[name]
        input_metadata.add(name, dtype=dtype, shape=shape[1:])

    # to provide at least iterations number of samples
    batches_number = (iterations // batch_size) + int(bool(iterations % batch_size))
    samples_number = batches_number * batch_size
    LOGGER.debug(
        f"Generating {batches_number} batches data with specs: {input_metadata} and value_ranges: {value_ranges}"
    )

    dataloader = DataLoader(iterations=samples_number, input_metadata=input_metadata, val_range=value_ranges)

    def _cast_input(name, value):
        target_type = dtypes[name]
        value = target_type.type(value)  # cast to target numpy dtype
        value = {"i": int(value), "u": int(value), "f": float(value)}[target_type.kind]  # cast to python primitive
        return value

    # FIXME: Workaround for DataLoader behavior
    #       For input shape tensors, i.e. inputs whose *value* describes a shape in the model, the
    #       provided shape will be used to populate the values of the inputs, rather than to determine
    #       their shape.
    #
    # WAR: when single value is generated override with min value from range
    def _create_content(name, data):
        casted = [_cast_input(name, x) for x in data.flatten().tolist()]
        if len(casted) == 1:
            x = value_ranges[name][0]
            casted = [_cast_input(name, x)]

        return casted

    data = {
        "data": [
            {
                name: {
                    "content": _create_content(name, data),
                    "shape": list(data.shape),
                }
                for name, data in feed_dict.items()
            }
            for idx, feed_dict in enumerate(dataloader)
            if idx < iterations
        ]
    }

    LOGGER.debug(f"Saving data of size {getsizeof(data)} bytes to {output_path}")
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    LOGGER.debug("File saved")
