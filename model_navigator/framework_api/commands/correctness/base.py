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
from typing import List, Optional, Tuple

import numpy
from polygraphy.backend.base import BaseRunner

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import Sample
from model_navigator.framework_api.exceptions import UserErrorContext
from model_navigator.framework_api.utils import DataObject, Framework, RuntimeProvider, sample_to_tuple
from model_navigator.model import Format


@dataclass
class Tolerance(DataObject):
    atol: float
    rtol: float


def get_assert_message(atol: float, rtol: float):
    return f"Current atol = {atol}, rtol = {rtol}, try to adjust tolerance values"


class CorrectnessBase(Command):
    def __init__(self, name: str, command_type: CommandType, target_format: Format, requires: Tuple[Command, ...] = ()):
        super().__init__(name=name, command_type=command_type, target_format=target_format, requires=requires)
        self.runtime_provider = RuntimeProvider.DEFAULT

    def _get_runners(self, **kwargs) -> Tuple[BaseRunner, BaseRunner]:
        raise NotImplementedError

    def __call__(
        self,
        correctness_samples: List[Sample],
        framework: Framework,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        **kwargs,
    ) -> Tolerance:

        base_runner, comp_runner = self._get_runners(
            correctness_samples=correctness_samples, framework=framework, atol=atol, rtol=rtol, **kwargs
        )
        atols = []
        rtols = []
        with base_runner, comp_runner:
            for sample in correctness_samples:
                with UserErrorContext():
                    original_output = base_runner.infer(sample)
                    comp_output = comp_runner.infer(sample)
                original_output, comp_output = sample_to_tuple(original_output), sample_to_tuple(comp_output)

                if atol and rtol:
                    assert len(original_output) == len(comp_output), get_assert_message(atol, rtol)
                    all_close_checks = [
                        numpy.allclose(tensor_A, tensor_B, atol=atol, rtol=rtol)
                        for tensor_A, tensor_B in zip(original_output, comp_output)
                    ]

                    assert all(all_close_checks), get_assert_message(atol, rtol)
                else:
                    per_output_atol = []
                    per_output_rtol = []
                    for o_out, c_out in zip(original_output, comp_output):
                        outputs_diff = numpy.fabs(o_out - c_out)
                        per_output_atol.append(numpy.max(outputs_diff))
                        per_output_rtol.append(numpy.mean(outputs_diff ** 2))
                    atols.append(numpy.max(per_output_atol))
                    rtols.append(numpy.mean(per_output_rtol))
        if not atol or not rtol:
            return Tolerance(atol=numpy.max(atols).item(), rtol=numpy.max(rtols).item())
        else:
            return Tolerance(atol, rtol)
