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


import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.environment_info import get_env, get_git_info
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import (
    DataObject,
    Framework,
    JitType,
    RuntimeProvider,
    Status,
    format_to_relative_model_path,
    get_package_path,
)
from model_navigator.model import Format

NAV_PACKAGE_FORMAT_VERSION = "0.1.0"


@dataclass
class RuntimeResults(DataObject):
    runtime: RuntimeProvider
    status: Status
    atol: Optional[float] = None
    rtol: Optional[float] = None
    performance: Optional[dict] = None
    err_msg: Optional[dict] = None


@dataclass
class ModelStatus(DataObject):
    format: Format
    path: Path
    runtime_results: List[RuntimeResults]
    torch_jit: Optional[JitType] = None
    precision: Optional[TensorRTPrecision] = None
    verified: bool = False


@dataclass
class NavigatorStatus(DataObject):
    format_version: str
    uuid: str
    git_info: Dict
    environment: Dict
    framework_navigator_config: Dict
    model_status: List[ModelStatus]


class PackageDescriptor:
    def __init__(self, pipelines: List[Pipeline], config: Config):
        self.pipelines = pipelines
        self.config = config

        model_status = []
        max_batch_size = None
        for pipeline in self.pipelines:
            for command in pipeline.commands:
                if command.command_type in (CommandType.EXPORT, CommandType.CONVERT):
                    runtime_results = []
                    if command.target_format == Format.ONNX:
                        runtimes = [RuntimeProvider.CPU.value, RuntimeProvider.CUDA.value, RuntimeProvider.TRT.value]
                    else:
                        runtimes = [RuntimeProvider.DEFAULT.value]

                    for runtime_provider in runtimes:
                        correctness_results = self._get_correctness_command_for_model(
                            commands=pipeline.commands,
                            format=command.target_format,
                            jit_type=command.target_jit_type,
                            precision=command.target_precision,
                            runtime_provider=runtime_provider,
                        )
                        performance_results = self._get_performance_command_for_model(
                            commands=pipeline.commands,
                            format=command.target_format,
                            jit_type=command.target_jit_type,
                            precision=command.target_precision,
                            runtime_provider=runtime_provider,
                        )

                        if (
                            command.status == Status.OK
                            and correctness_results
                            and correctness_results.output
                            and correctness_results.status == Status.OK
                            and performance_results
                            and performance_results.output
                            and performance_results.status == Status.OK
                        ):
                            status = Status.OK
                            err_msg = None
                            atol = correctness_results.output.atol
                            rtol = correctness_results.output.rtol
                            perf = performance_results.output
                        else:
                            status = Status.FAIL
                            err_msg = self.get_err_msg(command, correctness_results, performance_results)
                            atol = None
                            rtol = None
                            perf = None

                        runtime_results.append(
                            RuntimeResults(
                                runtime=runtime_provider,
                                status=status,
                                atol=atol,
                                rtol=rtol,
                                performance=perf,
                                err_msg=err_msg,
                            )
                        )

                    model_status.append(
                        ModelStatus(
                            format=command.target_format,
                            path=command.output,
                            torch_jit=command.target_jit_type,
                            precision=command.target_precision,
                            runtime_results=runtime_results,
                        )
                    )
                elif command.command_type == CommandType.FETCH_MODEL_INPUT:
                    if command.status == Status.OK:
                        max_batch_size = dict(zip(command.get_output_name(), command.output))["max_batch_size"]

        if not config.disable_git_info:
            git_info = get_git_info()
        else:
            git_info = None
        config = self.config.to_dict(
            filter_fields=[
                "model",
                "dataloader",
                "workdir",
                "keep_workdir",
                "override_workdir",
                "input_metadata",
                "output_metadata",
                "forward_kw_names",
                "disable_git_info",
                "zip_package",
            ],
            parse=True,
        )
        config["dataloader_batch_size"] = max_batch_size
        self.navigator_status = NavigatorStatus(
            uuid=str(uuid.uuid1()),
            format_version=NAV_PACKAGE_FORMAT_VERSION,
            git_info=git_info,
            environment=get_env(),
            framework_navigator_config=config,
            model_status=model_status,
        )

        self._create_status_file()
        if self.config.zip_package:
            self._zip_package()

        LOGGER.warning(
            "Initially models are not verified. Validate exported models and use "
            "PackageDescriptor.set_verified(format, jit_type, precision) method to set models as verified."
        )

    @staticmethod
    def get_err_msg(command, correctness_results, performance_results):
        err_msg = {}
        if command:
            if command.err_msg:
                err_msg[command.command_type.value] = command.err_msg
        if correctness_results:
            if correctness_results.err_msg:
                err_msg[correctness_results.command_type.value] = correctness_results.err_msg
        if performance_results is not None:
            if performance_results.err_msg:
                err_msg[performance_results.command_type.value] = performance_results.err_msg
        return err_msg

    @staticmethod
    def _get_correctness_command_for_model(
        commands: List[Command],
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
        runtime_provider: Optional[RuntimeProvider] = None,
    ):
        for command in commands:
            if (
                command.command_type == CommandType.CORRECTNESS
                and command.target_format == format
                and command.target_jit_type == jit_type
                and command.target_precision == precision
                and command.runtime_provider == runtime_provider
            ):
                return command
        return None

    @staticmethod
    def _get_performance_command_for_model(
        commands: List[Command],
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
        runtime_provider: Optional[RuntimeProvider] = None,
    ):
        for command in commands:
            if (
                command.command_type == CommandType.PERFORMANCE
                and command.target_format == format
                and command.target_jit_type == jit_type
                and command.target_precision == precision
                and command.runtime_provider == runtime_provider
            ):
                return command
        return None

    def _zip_package(self):
        archive_path = self.config.workdir / f"{self.config.model_name}.nav"
        nav_package_path = self.config.workdir / f"{self.config.model_name}.nav"
        LOGGER.info(f"Creating zip archive from {self.config.model_name}.nav")
        shutil.make_archive(archive_path.as_posix(), "zip", nav_package_path)

    def _create_status_file(self):
        import yaml

        with open(get_package_path(self.config.workdir, self.config.model_name) / "status.yaml", "w") as f:
            yaml.safe_dump(self.navigator_status.to_dict(parse=True), f, sort_keys=False)

    def _delete_status_file(self):
        status_file_path = get_package_path(self.config.workdir, self.config.model_name) / "status.yaml"
        status_file_path.unlink()

    def _load_model(self, model_path, framework: Framework, format: Format):
        LOGGER.info(f"Loading model from path: {model_path}")
        if framework == Framework.PYT:
            if format == Format.ONNX:
                import onnx  # pytype: disable=import-error

                return onnx.load(model_path)
            else:
                import torch  # pytype: disable=import-error

                return torch.jit.load(model_path)
        else:
            import tensorflow  # pytype: disable=import-error

            return tensorflow.keras.models.load_model(model_path)

    def set_verified(
        self, format: Format, jit_type: Optional[JitType] = None, precision: Optional[TensorRTPrecision] = None
    ):
        for model_status in self.navigator_status.model_status:
            if (
                model_status.format == format
                and model_status.torch_jit == jit_type
                and model_status.precision == precision
            ):
                model_status.verified = True
                self._delete_status_file()
                self._create_status_file()
                return

    def get_status(
        self,
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
        runtime_provider: Optional[RuntimeProvider] = RuntimeProvider.DEFAULT,
    ) -> bool:
        """Return status (True or False) of export operation for particular format, jit_type,
        precision and runtime_provider."""
        status = False
        for model_status in self.navigator_status.model_status:
            if (
                model_status.format == format
                and model_status.torch_jit == jit_type
                and model_status.precision == precision
            ):
                for runtime_results in model_status.runtime_results:
                    if runtime_provider == runtime_results.runtime and runtime_results.status == Status.OK:
                        status = True
        return status

    def get_formats_status(self) -> Dict:
        """Return dictionary of pairs Format : Bool. True for successful exports, False for failed exports."""
        results = {}
        for model_status in self.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                key = model_status.format.value
                if model_status.torch_jit:
                    key += f"-{model_status.torch_jit.value}"
                if model_status.precision:
                    key += f"-{model_status.precision.value}"
                key += f"-{runtime_results.runtime.value}"
                results[key] = runtime_results.status
        return results

    def get_formats_performance(self) -> Dict:
        """Return dictionary of pairs Format : Float with information about the median latency [ms] for each format."""
        results = {}
        for model_status in self.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                key = model_status.format.value
                if model_status.torch_jit:
                    key += f"-{model_status.torch_jit.value}"
                if model_status.precision:
                    key += f"-{model_status.precision.value}"
                key += f"-{runtime_results.runtime.value}"
                results[key] = runtime_results.performance
        return results

    def get_model(
        self, format: Format, jit_type: Optional[JitType] = None, precision: Optional[TensorRTPrecision] = None
    ):
        """Load exported model for given format, jit_type and precision and return model object"""
        model_path = get_package_path(
            workdir=self.config.workdir, model_name=self.config.model_name
        ) / format_to_relative_model_path(format=format, jit_type=jit_type, precision=precision)
        if model_path.exists():
            return self._load_model(model_path=model_path, framework=self.config.framework, format=format)
        else:
            return None
