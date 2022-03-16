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

import traceback
import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from inspect import getfullargspec
from typing import Any, Optional, Tuple

import typing_inspect

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.errors import ExternalError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import DataObject, JitType, Parameter, RuntimeProvider, Status
from model_navigator.model import Format


@dataclass
class Tolerance(DataObject):
    atol: float
    rtol: float


@dataclass
class Performance(DataObject):
    batch_size: int
    latency: float  # ms
    throughput: float  # infer / sec


class CommandType(Parameter):
    CONVERT = "convert"
    EXPORT = "export"
    CORRECTNESS = "correctness"
    GEN_CONFIG = "generate configuration"
    DUMP_MODEL_INPUT = "dump-model-input"
    DUMP_MODEL_OUTPUT = "dump-model-output"
    FETCH_MODEL_INPUT = "fetch-model-input"
    CUSTOM = "custom"
    PERFORMANCE = "performance"


@dataclass
class CommandResults(DataObject):
    name: str
    command_type: CommandType
    target_format: Format
    status: Status
    target_jit_type: Optional[JitType]
    target_precision: Optional[TensorRTPrecision]
    runtime_provider: Optional[RuntimeProvider]
    missing_params: Optional[dict]
    output: Any
    err_msg: Optional[str] = None

    def get_formatted_command_details(
        self,
    ):
        cmd_name_and_details = f"[{self.status.value:^4}] {self.name}"
        if self.target_jit_type:
            cmd_name_and_details += f" {self.target_jit_type}"
        if self.target_precision:
            cmd_name_and_details += f" {self.target_precision}"
        if self.runtime_provider:
            cmd_name_and_details += f" {self.runtime_provider}"
        return cmd_name_and_details


class Command(metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        command_type: CommandType,
        target_format: Optional[Format] = None,
        requires: Tuple["Command", ...] = (),
    ):
        self.name = name
        self.command_type = command_type
        self.target_format = target_format
        self.status = Status.INITIALIZED
        self._requires = requires

    def __getattr__(self, item):
        return None

    def transform(self, **kwargs) -> CommandResults:
        status = self._validate(**kwargs)
        missing_params = {}
        results = None
        err_msg = None

        if self._check_requires():
            try:
                if status == Status.OK:
                    results = self.__call__(**kwargs)

                else:
                    missing_params = self._get_missing_params(**kwargs)
            except Exception as e:

                status = Status.FAIL
                err_msg = str(e)

                LOGGER.error(f"{type(e).__name__} raised.")
                if isinstance(e, ExternalError):
                    LOGGER.warning(
                        "External errors are usually caused by incompatibilites between the model and the target formats and/or runtimes."
                    )
                LOGGER.error(traceback.format_exc())
                import os

                if "NAV_DEBUG" in os.environ:
                    raise e
                else:
                    LOGGER.info("You can disable error suppression for debugging with flag NAV_DEBUG=1")

        else:
            status = Status.SKIPPED

        self.status = status
        return CommandResults(
            name=self.name,
            status=self.status,
            command_type=self.command_type,
            target_format=self.target_format,
            target_jit_type=self.target_jit_type,
            target_precision=self.target_precision,
            missing_params=missing_params,
            runtime_provider=self.runtime_provider,
            err_msg=err_msg,
            output=results,
        )

    def _check_requires(self):
        for req in self._requires:
            if req.status != Status.OK:
                LOGGER.warning(f"This command requires '{req.name}' but it's status is {req.status.value}. Skipping...")
                return False
        return True

    @staticmethod
    def get_output_name() -> Optional[str]:
        return None

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @staticmethod
    def _is_param_optional(param_annotation):
        return typing_inspect.get_origin(param_annotation) is typing.Union and type(None) in typing_inspect.get_args(
            param_annotation
        )

    def _validate(self, **kwargs):
        required_params = getfullargspec(self.__call__).args[1:]
        provided_params = kwargs
        for param in required_params:
            annotations = getfullargspec(self.__call__).annotations
            # If parameter is not optional and parameter is None return Fail
            if not self._is_param_optional(annotations.get(param)) and provided_params.get(param) is None:
                return Status.FAIL
        return Status.OK

    def _get_missing_params(self, **kwargs):
        required_params = getfullargspec(self.__call__).args[1:]
        provided_params = kwargs
        missing_params = {}
        for param in required_params:
            annotations = getfullargspec(self.__call__).annotations
            # If parameter is not optional and parameter is None parameter is missing
            if not self._is_param_optional(annotations.get(param)) and provided_params.get(param) is None:
                missing_params[
                    param
                ] = f"{getfullargspec(self.__call__).annotations[param]} = {provided_params.get(param)}"
        return missing_params
