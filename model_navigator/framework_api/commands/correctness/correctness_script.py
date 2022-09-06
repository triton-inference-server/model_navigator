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
import pathlib
import sys
from typing import Dict, List, Optional

import fire
import numpy as np
from polygraphy.comparator import util as comp_util

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.correctness import Tolerance, TolerancePerOutputName
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.runners.runner_manager import RunnerManager
from model_navigator.framework_api.utils import Format, JitType, RuntimeProvider, load_samples


def correctness(
    model_name: str,
    output_names: List[str],
    batch_dim: int,
    results_path: str,
    format: str,
    precision: str,
    jit_type: str,
    runtime: str,
    enable_xla: bool,
    jit_compile: bool,
    runner_manager_dict: Dict,
    navigator_workdir: Optional[str] = None,
):
    if not navigator_workdir:
        navigator_workdir = pathlib.Path.cwd()
    navigator_workdir = pathlib.Path(navigator_workdir)

    correctness_samples = load_samples("correctness_samples", navigator_workdir, batch_dim)
    correctness_samples_output = load_samples("correctness_samples_output", navigator_workdir, batch_dim)

    runner = RunnerManager.from_dict(runner_manager_dict).get_runner(
        workdir=navigator_workdir,
        format=Format(format),
        jit_type=JitType(jit_type) if jit_type else None,
        precision=TensorRTPrecision(precision) if precision else None,
        runtime=RuntimeProvider(runtime) if runtime else None,
        enable_xla=enable_xla,
        jit_compile=jit_compile,
    )

    per_output_tolerance = TolerancePerOutputName({name: Tolerance(0.0, 0.0) for name in output_names})
    with runner:
        for sample, original_output in zip(correctness_samples, correctness_samples_output):
            comp_output = runner.infer(sample)

            is_len_valid = len(original_output) == len(comp_output)
            if not is_len_valid:
                LOGGER.error("Original model output length is different from exported model output")
                sys.exit(1)

            for name in output_names:
                if any(np.isnan(comp_output[name]).flatten()):
                    LOGGER.error("Comparison output contains NaN")
                    sys.exit(1)

                if any(np.isinf(comp_output[name]).flatten()):
                    LOGGER.error("Comparison output contains inf")
                    sys.exit(1)

                out0, out1 = original_output[name], comp_output[name]
                absdiff = np.abs(out0 - out1)
                absout1 = np.abs(out1)

                reldiff = absdiff / absout1
                max_reldiff = comp_util.compute_max(reldiff)
                max_absdiff = comp_util.compute_max(absdiff)

                if max_absdiff > per_output_tolerance[name].atol:
                    per_output_tolerance[name].atol = float(max_absdiff)
                if max_reldiff > per_output_tolerance[name].rtol:
                    per_output_tolerance[name].rtol = float(max_reldiff)

    results_path = pathlib.Path(results_path)
    with results_path.open("w") as f:
        json.dump(per_output_tolerance.to_json(), f)


if __name__ == "__main__":
    fire.Fire(correctness)
