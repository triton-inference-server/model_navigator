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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy
from polygraphy.backend.base import BaseRunner
from polygraphy.comparator import util as comp_util

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import DataObject, Sample, TensorMetadata
from model_navigator.framework_api.exceptions import UserErrorContext
from model_navigator.framework_api.utils import Framework, Status, format2runtimes, sample_to_tuple
from model_navigator.model import Format


@dataclass
class Tolerance(DataObject):
    atol: float
    rtol: float

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(
            atol=dict["atol"],
            rtol=dict["rtol"],
        )


class TolerancePerOutputName(Dict[str, Tolerance]):
    def to_json(self):
        return [{"output_name": name, **tol.to_dict(parse=True)} for name, tol in self.items()]

    @classmethod
    def from_json(cls, data: List):
        tol_per_out = cls()
        for tol in data:
            tol_per_out[tol["output_name"]] = Tolerance.from_dict(tol)
        return tol_per_out


def get_assert_message(atol: float, rtol: float):
    return f"Current atol = {atol}, rtol = {rtol}, try to adjust tolerance values"


class CorrectnessBase(Command):
    def __init__(self, name: str, command_type: CommandType, target_format: Format, requires: Tuple[Command, ...] = ()):
        super().__init__(name=name, command_type=command_type, target_format=target_format, requires=requires)
        self.runtime_provider = format2runtimes(target_format)[0]

    def _get_runner(self, **kwargs) -> BaseRunner:
        raise NotImplementedError

    def __call__(
        self,
        correctness_samples: List[Sample],
        correctness_samples_output: List[Sample],
        framework: Framework,
        output_metadata: TensorMetadata,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        **kwargs,
    ) -> TolerancePerOutputName:

        runner = self._get_runner(
            correctness_samples=correctness_samples, framework=framework, output_metadata=output_metadata, **kwargs
        )
        output_names = [v.name for v in output_metadata.values()]
        per_output_tolerance = TolerancePerOutputName({name: Tolerance(0.0, 0.0) for name in output_names})
        with runner:
            for sample, original_output in zip(correctness_samples, correctness_samples_output):
                with UserErrorContext():
                    comp_output = runner.infer(sample)
                original_output, comp_output = sample_to_tuple(original_output), sample_to_tuple(comp_output)

                is_len_valid = len(original_output) == len(comp_output)
                assert is_len_valid, "Original model output length is different from exported model output"

                for out0, out1, name in zip(original_output, comp_output, output_names):
                    absdiff = numpy.abs(out0 - out1)
                    absout1 = numpy.abs(out1)

                    reldiff = absdiff / absout1
                    max_reldiff = comp_util.compute_max(reldiff)
                    max_absdiff = comp_util.compute_max(absdiff)

                    if max_absdiff > per_output_tolerance[name].atol:
                        per_output_tolerance[name].atol = float(max_absdiff)
                    if max_reldiff > per_output_tolerance[name].rtol:
                        per_output_tolerance[name].rtol = float(max_reldiff)

                    def is_diff_within_tol(diff, tol):
                        return not numpy.isnan(diff) and diff <= tol

                    if is_len_valid and atol is not None:
                        if not is_diff_within_tol(max_absdiff, atol):
                            self.status = Status.FAIL
                            self.err_msg = get_assert_message(atol, rtol)
                    if is_len_valid and rtol is not None:
                        if not is_diff_within_tol(max_reldiff, rtol):
                            self.status = Status.FAIL
                            self.err_msg = get_assert_message(atol, rtol)
        return per_output_tolerance
