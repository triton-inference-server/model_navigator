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
from pathlib import Path
from typing import Dict, Optional

import fire

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.performance import Profiler
from model_navigator.framework_api.commands.performance.performance import ProfilerConfig
from model_navigator.framework_api.runners.runner_manager import RunnerManager
from model_navigator.framework_api.utils import Format, JitType, RuntimeProvider, load_samples


def profile(
    workdir: str,
    model_name: str,
    package_path: str,
    batch_dim: int,
    results_path: str,
    format: str,
    precision: str,
    jit_type: str,
    runtime: str,
    profiler_config: Dict,
    max_batch_size: Optional[int],
    enable_xla: bool,
    jit_compile: bool,
    runner_manager_dict: Dict,
):

    profiling_sample = load_samples("profiling_sample", package_path, batch_dim)
    results_path = Path(results_path)

    runner = RunnerManager.from_dict(runner_manager_dict).get_runner(
        workdir=Path(workdir),
        model_name=model_name,
        format=Format(format),
        jit_type=JitType(jit_type) if jit_type else None,
        precision=TensorRTPrecision(precision) if precision else None,
        runtime=RuntimeProvider(runtime) if runtime else None,
        enable_xla=enable_xla,
        jit_compile=jit_compile,
    )

    results = Profiler(
        runner, profiling_sample, ProfilerConfig.from_dict(profiler_config), batch_dim, max_batch_size
    ).run()

    with results_path.open("w") as f:
        json.dump([res.to_dict() for res in results], f)


if __name__ == "__main__":
    fire.Fire(profile)
