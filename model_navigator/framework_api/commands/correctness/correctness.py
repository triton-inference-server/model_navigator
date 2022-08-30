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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import DataObject, TensorMetadata
from model_navigator.framework_api.exceptions import ExecutionContext
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.runners.runner_manager import RunnerManager
from model_navigator.framework_api.utils import (
    JitType,
    RuntimeProvider,
    Status,
    format_to_relative_model_path,
    get_package_path,
    parse_kwargs_to_cmd,
)
from model_navigator.model import Format

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


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


class Correctness(Command):
    def __init__(
        self,
        name: str,
        target_format: Format,
        requires: Tuple[Command, ...] = (),
        target_jit_type: Optional[JitType] = None,
        target_precision: Optional[TensorRTPrecision] = None,
        runtime_provider: Optional[RuntimeProvider] = None,
        enable_xla: Optional[bool] = None,
        jit_compile: Optional[bool] = None,
    ):
        super().__init__(
            name=name, command_type=CommandType.CORRECTNESS, target_format=target_format, requires=requires
        )
        self.target_jit_type = target_jit_type
        self.target_precision = target_precision
        self.runtime_provider = runtime_provider
        self.enable_xla = enable_xla
        self.jit_compile = jit_compile

    def _update_package_descriptor(self, package_descriptor: "PackageDescriptor", **kwargs) -> None:
        runtime_results = package_descriptor.get_runtime_results(
            format=self.target_format,
            jit_type=self.target_jit_type,
            precision=self.target_precision,
            runtime_provider=self.runtime_provider,
            enable_xla=self.enable_xla,
            jit_compile=self.jit_compile,
        )
        if runtime_results.status == Status.OK:
            if self.status == Status.OK:
                runtime_results.tolerance = self.output
            else:
                runtime_results.status = self.status
                runtime_results.err_msg[self.command_type.value] = self.err_msg

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: str,
        batch_dim: Optional[int],
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        **kwargs,
    ) -> TolerancePerOutputName:
        LOGGER.info(f"Correctness test for: {self.target_format} {self.runtime_provider}started.")

        model_path = get_package_path(workdir=workdir, model_name=model_name) / format_to_relative_model_path(
            format=self.target_format,
            jit_type=self.target_jit_type,
            precision=self.target_precision,
            enable_xla=self.enable_xla,
            jit_compile=self.jit_compile,
        )
        model_dir = model_path.parent
        output_names = list(output_metadata.keys())

        runner_manager = RunnerManager(input_metadata, output_metadata, target_device)

        with ExecutionContext(
            model_dir / "reproduce_correctness.py", model_dir / "reproduce_correctness.sh"
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "workdir": workdir.as_posix(),
                "model_name": model_name,
                "output_names": output_names,
                "package_path": get_package_path(workdir, model_name).as_posix(),
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "format": self.target_format.value,
                "precision": self.target_precision.value if self.target_precision else None,
                "jit_type": self.target_jit_type.value if self.target_jit_type else None,
                "runtime": self.runtime_provider.value if self.runtime_provider else None,
                "enable_xla": self.enable_xla,
                "jit_compile": self.jit_compile,
                "runner_manager_dict": runner_manager.to_dict(parse=True),
            }

            args = parse_kwargs_to_cmd(kwargs, (list, dict, tuple))

            from model_navigator.framework_api.commands.correctness import correctness_script

            context.execute_external_runtime_script(correctness_script.__file__, args)
            per_output_tolerance = TolerancePerOutputName.from_json(json.load(temp_file))

        def is_diff_within_tol(diff, tol):
            return not numpy.isnan(diff) and diff <= tol

        for name in output_names:
            if atol is not None:
                if not is_diff_within_tol(per_output_tolerance[name].atol, atol):
                    self.status = Status.FAIL
                    self.err_msg = get_assert_message(atol, rtol)
            if rtol is not None:
                if not is_diff_within_tol(per_output_tolerance[name].rtol, rtol):
                    self.status = Status.FAIL
                    self.err_msg = get_assert_message(atol, rtol)

        return per_output_tolerance
