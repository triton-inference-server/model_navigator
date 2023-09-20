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
"""Script for running correctness tests on a runner."""

import pathlib
from typing import Dict, Optional

import fire

from model_navigator.core.dataloader import load_samples
from model_navigator.core.tensor import TensorMetadata
from model_navigator.runners.registry import get_runner


def load_model(
    batch_dim: int,
    runner_name: str,
    input_metadata: Dict,
    output_metadata: Dict,
    model_path: str,
    navigator_workspace: Optional[str] = None,
) -> None:
    """Run correcntess tests.

    Args:
        batch_dim (int): Batch dimension.
        runner_name (str): Name of the model's runner.
        input_metadata (Dict): Input metadata.
        output_metadata (Dict): Output metadata.
        model_path (str): Path to the model.
        navigator_workspace (Optional[str], optional): Path of the Model Navigator workspace.
            When None use current workdir. Defaults to None.
    """
    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    model = navigator_workspace / model_path

    input_metadata = TensorMetadata.from_json(input_metadata)
    output_metadata = TensorMetadata.from_json(output_metadata)
    runner = get_runner(runner_name)(
        model=model,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
    )  # pytype: disable=not-instantiable

    with runner:
        runner.infer(profiling_sample)


if __name__ == "__main__":
    fire.Fire(load_model)
