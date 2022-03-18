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
class ModelStatus(DataObject):
    format: Format
    path: Path
    status: dict
    tolerance: dict
    torch_jit: Optional[JitType] = None
    precision: Optional[TensorRTPrecision] = None
    provider: Optional[RuntimeProvider] = None
    verified: bool = False
    performance: Optional[dict] = None
    err_msg: Optional[dict] = None


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

        # pipeline_results to navigator_status
        model_status = []
        max_batch_size = None
        for pipeline in self.pipelines:
            for command in pipeline.commands:
                if command.command_type in (CommandType.EXPORT, CommandType.CONVERT):

                    status_per_provider = {}
                    err_mgs_per_provider = {}
                    correctness_results_per_provider = {}
                    performance_results_per_provider = {}

                    if command.target_format == Format.ONNX:
                        runtimes = [RuntimeProvider.CPU.value, RuntimeProvider.CUDA.value, RuntimeProvider.TRT.value]
                    else:
                        runtimes = [RuntimeProvider.DEFAULT.value]

                    for provider in runtimes:
                        correctness_results = self._get_correctness_command_for_model(
                            commands=pipeline.commands,
                            format=command.target_format,
                            jit_type=command.target_jit_type,
                            precision=command.target_precision,
                            runtime_provider=provider,
                        )
                        if correctness_results is not None:
                            correctness_results_per_provider[provider] = correctness_results.output

                        if (
                            command.status == Status.OK
                            and correctness_results is not None
                            and correctness_results.status == Status.OK
                        ):
                            status_per_provider[provider] = Status.OK
                            err_mgs_per_provider[provider] = None
                        else:
                            status_per_provider[provider] = Status.FAIL
                            err_mgs_per_provider[provider] = command.err_msg

                        performance_results = self._get_performance_command_for_model(
                            commands=pipeline.commands,
                            format=command.target_format,
                            jit_type=command.target_jit_type,
                            precision=command.target_precision,
                            runtime_provider=provider,
                        )
                        if performance_results is not None:
                            performance_results_per_provider[provider] = performance_results.output

                    if all(msg is None for msg in err_mgs_per_provider.values()):
                        err_mgs_per_provider = None

                    model_status.append(
                        ModelStatus(
                            format=command.target_format,
                            path=command.output,
                            status=status_per_provider,
                            tolerance=correctness_results_per_provider,
                            torch_jit=command.target_jit_type,
                            precision=command.target_precision,
                            performance=performance_results_per_provider,
                            err_msg=err_mgs_per_provider,
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
                and model_status.status[runtime_provider] == Status.OK
            ):
                status = True
        return status

    def get_formats_status(self) -> Dict:
        """Return dictionary of pairs Format : Bool. True for successful exports, False for failed exports."""
        results = {}
        for model_status in self.navigator_status.model_status:
            key = model_status.format.value
            if model_status.torch_jit:
                key += f"-{model_status.torch_jit.value}"
            if model_status.precision:
                key += f"-{model_status.precision.value}"
            results[key] = model_status.status
        return results

    def get_formats_performance(self) -> Dict:
        """Return dictionary of pairs Format : Float with information about the median latency [ms] for each format."""
        results = {}
        for model_status in self.navigator_status.model_status:
            key = model_status.format.value
            if model_status.torch_jit:
                key += f"-{model_status.torch_jit.value}"
            if model_status.precision:
                key += f"-{model_status.precision.value}"
            results[key] = model_status.performance
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
