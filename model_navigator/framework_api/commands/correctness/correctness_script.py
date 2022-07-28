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
from typing import Dict, List

import fire
import numpy as np
from polygraphy.comparator import util as comp_util

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.correctness import Tolerance, TolerancePerOutputName
from model_navigator.framework_api.runners.runner_manager import RunnerManager
from model_navigator.framework_api.utils import Format, JitType, RuntimeProvider, load_samples


def correctness(
    workdir: str,
    model_name: str,
    output_names: List[str],
    package_path: str,
    batch_dim: int,
    results_path: str,
    format: str,
    precision: str,
    jit_type: str,
    runtime: str,
    runner_manager_dict: Dict,
):

    correctness_samples = load_samples("correctness_samples", package_path, batch_dim)
    correctness_samples_output = load_samples("correctness_samples_output", package_path, batch_dim)
    results_path = Path(results_path)

    runner = RunnerManager.from_dict(runner_manager_dict).get_runner(
        workdir=Path(workdir),
        model_name=model_name,
        format=Format(format),
        jit_type=JitType(jit_type) if jit_type else None,
        precision=TensorRTPrecision(precision) if precision else None,
        runtime=RuntimeProvider(runtime) if runtime else None,
    )

    per_output_tolerance = TolerancePerOutputName({name: Tolerance(0.0, 0.0) for name in output_names})
    with runner:
        for sample, original_output in zip(correctness_samples, correctness_samples_output):
            comp_output = runner.infer(sample)

            is_len_valid = len(original_output) == len(comp_output)
            assert is_len_valid, "Original model output length is different from exported model output"

            for name in output_names:
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

    with results_path.open("w") as f:
        json.dump(per_output_tolerance.to_json(), f)


if __name__ == "__main__":
    fire.Fire(correctness)
