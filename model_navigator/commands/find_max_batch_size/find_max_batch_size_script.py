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
from typing import Dict, Optional

import fire

from model_navigator.api.config import OptimizationProfile
from model_navigator.commands.find_max_batch_size.find_max_batch_size import MaxBatchSizeFinder
from model_navigator.core.dataloader import load_samples
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.runners.registry import get_runner


def find_max_batch_size(
    batch_dim: int,
    results_path: str,
    optimization_profile: Dict,
    model_path: str,
    runner_name: str,
    input_metadata: Dict,
    output_metadata: Dict,
    navigator_workspace: Optional[str] = None,
    runner_config: Optional[Dict] = None,
) -> None:
    """Find device maximum batch size.

    Args:
        batch_dim: Batch dimension.
        results_path: Output results path.
        optimization_profile: Optimization profile used during conversion and profiling.
        model_path: Path to the model.
        runner_name: Name of the model's runner.
        input_metadata: Input metadata.
        output_metadata: Output metadata.
        navigator_workspace: Model Navigator workspace path. When None use current workdir. Defaults to None.
        runner_config: Additional runner configuration.
    """
    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    results_path = pathlib.Path(results_path)

    if runner_config is None:
        runner_config = {}

    runner = get_runner(runner_name)(
        model=navigator_workspace / model_path,
        input_metadata=TensorMetadata.from_json(input_metadata),
        output_metadata=TensorMetadata.from_json(output_metadata),
        disable_fallback=False,
        enable_timer=True,
        **runner_config,
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
