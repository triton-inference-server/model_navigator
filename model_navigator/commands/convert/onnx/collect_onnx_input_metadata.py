# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Script for obtaining ONNX metadata from model."""

import json
import pathlib
from typing import Dict

import fire

from model_navigator.core.tensor import TensorMetadata
from model_navigator.runners.onnx import OnnxrtCPURunner


def collect_onnx_input_metadata(
    model_path: str,
    input_metadata: Dict,
    output_metadata: Dict,
    results_path: str,
) -> None:
    """Collect input metadata from onnx model.

    Args:
        model_path (str): Path to onnx the model.
        input_metadata (Dict): Input metadata.
        output_metadata (Dict): Output metadata.
        results_path (str): Path where metadata file is stored
    """
    onnx_runner = OnnxrtCPURunner(
        model=pathlib.Path(model_path),
        input_metadata=TensorMetadata.from_json(input_metadata),
        output_metadata=TensorMetadata.from_json(output_metadata),
    )
    with onnx_runner:
        obtained_metadata = onnx_runner.get_onnx_input_metadata()

    with open(results_path, "w") as fp:
        json.dump(obtained_metadata.to_json(), fp)


if __name__ == "__main__":
    fire.Fire(collect_onnx_input_metadata)
