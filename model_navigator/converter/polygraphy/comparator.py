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
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_TOLERANCE_ATOL = 1e-5
DEFAULT_TOLERANCE_RTOL = 1e-5


@dataclass
class OutputErrorStat:
    out0: np.ndarray
    out1: np.ndarray

    @property
    def absdiff(self):
        if np.issubdtype(self.out0.dtype, np.bool_) and np.issubdtype(self.out1.dtype, np.bool_):
            absdiff = np.logical_xor(self.out0, self.out1)
        else:
            absdiff = np.abs(self.out0 - self.out1)
        return absdiff

    @property
    def reldiff(self):
        _reldiff = self.absdiff / np.abs(self.out1)
        _reldiff[self.absdiff == 0] = 0
        return _reldiff

    def min_out(self):
        return np.amin([np.amin(self.out0), np.amin(self.out1)])

    def max_out(self):
        return np.amax([np.amax(self.out0), np.amax(self.out1)])

    def max_absdiff(self):
        return np.amax(self.absdiff)

    def max_reldiff(self):
        return np.amax(self.reldiff)

    def mean_absdiff(self):
        return np.mean(self.absdiff)

    def mean_reldiff(self):
        return np.mean(self.reldiff)

    def std_absdiff(self):
        return np.std(self.absdiff)

    def std_reldiff(self):
        return np.std(self.reldiff)


def _get_recommended_tolerance_params(iter_result0, iter_result1, std_coeff: int = 2):
    atol = {}
    rtol = {}
    for out0_name, output0 in iter_result0.items():
        output1 = iter_result1[out0_name]
        output_stat = OutputErrorStat(out0=output0, out1=output1)

        # heuristic: atol = <absdiff mean> + <std_coeff> * <absdiff std_dev>
        output_atol = output_stat.mean_absdiff() + std_coeff * output_stat.std_absdiff()

        # for those elements which abs diff values are above calculated atol - get max reldiff
        above_atol_idx = np.where(output_stat.absdiff >= output_atol)
        rtol_to_check = output_stat.reldiff[above_atol_idx]
        output_rtol = np.amax(rtol_to_check) if rtol_to_check.size > 0 else DEFAULT_TOLERANCE_RTOL

        atol[out0_name] = output_atol
        rtol[out0_name] = output_rtol

    return atol, rtol


def _get_values_range(iter_result0, iter_result1):
    stats = {
        out_name: OutputErrorStat(out0=iter_result0[out_name], out1=iter_result1[out_name]) for out_name in iter_result0
    }
    return {name: (stat.min_out(), stat.max_out()) for name, stat in stats.items()}


class ToleranceParameterHelper:
    def __init__(self, comparator_inputs_path: Path, comparator_outputs_path: Path):
        self._comparator_inputs_path = comparator_inputs_path
        self._comparator_outputs_path = comparator_outputs_path

    def get_tolerance_parameters(self):
        from polygraphy.comparator import RunResults

        MAX_RTOL = 0.05  # max 5% difference
        MAX_REL_COMPARING_ATOL_TO_MAX_ABS_OUTPUT = 0.01  # atol can be max 1% of max of abs of all outputs

        run_results = RunResults.load(self._comparator_outputs_path)
        comparisons = [(i, i + 1) for i in range(len(run_results) - 1)]

        value_ranges_per_output = self.get_outputs_value_ranges()
        max_value_per_output = {
            name: np.amax(np.abs(value_range)) for name, value_range in value_ranges_per_output.items()
        }

        atol_final, rtol_final = None, None
        std_coeff = 0

        while atol_final is None:
            atol_all_iterations = {}
            rtol_all_iterations = {}
            for runner0_index, runner1_index in comparisons:
                # run_results is list of tuples (<name>, <results>)
                (_, results0), (_, results1) = run_results[runner0_index], run_results[runner1_index]
                for result0, result1 in zip(results0, results1):
                    atol, rtol = _get_recommended_tolerance_params(result0, result1, std_coeff=std_coeff)
                    for name, tol in atol.items():
                        atol_all_iterations.setdefault(name, []).append(tol)
                        rtol_all_iterations.setdefault(name, []).append(rtol[name])

            def fractional_ceil(a, precision=0):
                return np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)

            # get max from calculated tolerance parameters
            atol = {name: fractional_ceil(np.amax(tols), 3) for name, tols in atol_all_iterations.items()}
            rtol = {name: fractional_ceil(np.amax(tols), 3) for name, tols in rtol_all_iterations.items()}

            def _check_atol_not_exceeds(current_atol_per_output):
                return current_atol_per_output is not None and all(
                    [
                        atol_value / max_value_per_output[output_name] < MAX_REL_COMPARING_ATOL_TO_MAX_ABS_OUTPUT
                        for output_name, atol_value in current_atol_per_output.items()
                    ]
                )

            # if exceeded threshold - stop search
            if not _check_atol_not_exceeds(atol):
                break

            # if atol don't exceeds - check if rtol is low enough
            if all(value < MAX_RTOL for value in rtol.values()):
                atol_final = atol
                rtol_final = rtol
                break

            # increase atol by another absdiff std_dev
            std_coeff += 1

        return atol_final, rtol_final

    def get_outputs_value_ranges(self):
        from polygraphy.comparator import RunResults

        run_results = RunResults.load(self._comparator_outputs_path)
        comparisons = [(i, i + 1) for i in range(len(run_results) - 1)]

        output_values_range_all_iterations = {}
        for runner0_index, runner1_index in comparisons:
            # run_results is list of tuples (<name>, <results>)
            (_, results0), (_, results1) = run_results[runner0_index], run_results[runner1_index]
            for result0, result1 in zip(results0, results1):
                values_range = _get_values_range(result0, result1)
                for name, value_range in values_range.items():
                    output_values_range_all_iterations.setdefault(name, []).append(value_range)

        out_values_ranges = {
            name: (np.amin([vr[0] for vr in value_ranges]), np.amax([vr[1] for vr in value_ranges]))
            for name, value_ranges in output_values_range_all_iterations.items()
        }

        return out_values_ranges
