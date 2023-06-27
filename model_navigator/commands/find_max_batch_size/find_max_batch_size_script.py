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
"""Script for finding device maximum batch size for a runner."""

import pathlib
from typing import Dict, List, Optional

import fire

from model_navigator.api.config import OptimizationProfile
from model_navigator.commands.find_max_batch_size.find_max_batch_size import MaxBatchSizeFinder
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.runners.registry import get_runner
from model_navigator.utils.dataloader import load_samples


def find_max_batch_size(
    batch_dim: int,
    results_path: str,
    optimization_profile: Dict,
    model_path: str,
    runner_name: str,
    input_metadata: List,
    output_metadata: List,
    navigator_workspace: Optional[str] = None,
) -> None:
    """Find device maximum batch size.

    Args:
        batch_dim (int): Batch dimension.
        results_path (str): Output results path.
        optimization_profile (Dict): Optimization profile used during conversion and profiling.
        model_path (str): Path to the model.
        runner_name (str): Name of the model's runner.
        input_metadata (List): Input metadata.
        output_metadata (List): Output metadata.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
    """
    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    results_path = pathlib.Path(results_path)

    runner = get_runner(runner_name)(
        model=navigator_workspace / model_path,
        input_metadata=TensorMetadata.from_json(input_metadata),
        output_metadata=TensorMetadata.from_json(output_metadata),
        disable_fallback=False,
    )  # pytype: disable=not-instantiable
    try:
        MaxBatchSizeFinder(
            profile=OptimizationProfile.from_dict(optimization_profile),
            batch_dim=batch_dim,
            results_path=results_path,
        ).run(
            runner=runner,
            profiling_sample=profiling_sample,
            sample_id=0,
        )
    except Exception as e:
        if results_path.is_file():
            LOGGER.info("Max batch size search finished with success.")
        else:
            raise e


if __name__ == "__main__":
    fire.Fire(find_max_batch_size)
