# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import CommandResults, CommandType, Performance, Tolerance
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.environment_info import get_env, get_git_info
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.pipelines.pipeline import PipelineResults
from model_navigator.framework_api.utils import (
    DataObject,
    Framework,
    JitType,
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
    status: Status
    tolerance: Tolerance
    torch_jit: Optional[JitType] = None
    precision: Optional[TensorRTPrecision] = None
    verified: bool = False
    performance: Optional[Performance] = None
    err_msg: Optional[str] = None


@dataclass
class NavigatorStatus(DataObject):
    format_version: str
    uuid: str
    git_info: Dict
    environment: Dict
    framework_navigator_config: Dict
    model_status: List[ModelStatus]


class PackageDescriptor:

    # TODO Cleanup checking correctness status for exported models.
    def _get_correctness_for_model(
        self,
        commands_results: List[CommandResults],
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
    ):
        for command_results in commands_results:
            if (
                command_results.command_type == CommandType.CORRECTNESS
                and command_results.target_format == format
                and command_results.target_jit_type == jit_type
                and command_results.target_precision == precision
            ):
                return command_results
        parameters = f"{format}, {precision}, {jit_type}"
        raise Exception(f"Correctness results not found for given parameters: {parameters}")

    def _get_performance_for_model(
        self,
        commands_results: List[CommandResults],
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
    ):
        for command_results in commands_results:
            if (
                command_results.command_type == CommandType.PERFORMANCE
                and command_results.target_format == format
                and command_results.target_jit_type == jit_type
                and command_results.target_precision == precision
            ):
                return command_results
        return None

    def __init__(self, pipelines_results: List[PipelineResults], config: Config):
        self.pipelines_results = pipelines_results
        self.config = config

        # pipeline_results to navigator_status
        model_status = []
        for pipeline_results in self.pipelines_results:

            for command_results in pipeline_results.commands_results:
                if command_results.command_type in (CommandType.EXPORT, CommandType.CONVERT):
                    correctness_results = self._get_correctness_for_model(
                        commands_results=pipeline_results.commands_results,
                        format=command_results.target_format,
                        jit_type=command_results.target_jit_type,
                        precision=command_results.target_precision,
                    )

                    if command_results.status == Status.OK and correctness_results.status == Status.OK:
                        status = Status.OK
                        err_msg = None
                    else:
                        status = Status.FAIL
                        err_msg = command_results.err_msg

                    performance_results = self._get_performance_for_model(
                        commands_results=pipeline_results.commands_results,
                        format=command_results.target_format,
                        jit_type=command_results.target_jit_type,
                        precision=command_results.target_precision,
                    )
                    if performance_results is not None:
                        performance_results = performance_results.output

                    model_status.append(
                        ModelStatus(
                            format=command_results.target_format,
                            path=command_results.output,
                            status=status,
                            tolerance=correctness_results.output,
                            torch_jit=command_results.target_jit_type,
                            precision=command_results.target_precision,
                            performance=performance_results,
                            err_msg=err_msg,
                        )
                    )

        self.navigator_status = NavigatorStatus(
            uuid=str(uuid.uuid1()),
            format_version=NAV_PACKAGE_FORMAT_VERSION,
            git_info=get_git_info(),
            environment=get_env(),
            framework_navigator_config=self.config.to_dict(
                filter_fields=[
                    "model",
                    "dataloader",
                    "workdir",
                    "keep_workdir",
                    "override_workdir",
                    "input_metadata",
                    "output_metadata",
                    "forward_kw_names",
                ],
                parse=True,
            ),
            model_status=model_status,
        )

        self._create_status_file()
        LOGGER.warning(
            "Initially models are not verified. Validate exported models and use "
            "PackageDescriptor.set_verified(format, jit_type, precision) method to set models as verified."
        )

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
        self, format: Format, jit_type: Optional[JitType] = None, precision: Optional[TensorRTPrecision] = None
    ) -> bool:
        """Return status (True or False) of export operation for particular format, jit_type and precision."""
        status = False
        for model_status in self.navigator_status.model_status:
            if (
                model_status.format == format
                and model_status.torch_jit == jit_type
                and model_status.precision == precision
                and model_status.status == Status.OK
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

    def get_formats_latency(self) -> Dict:
        """Return dictionary of pairs Format : Float with information about the median latency [ms] for each format."""
        results = {}
        for model_status in self.navigator_status.model_status:
            key = model_status.format.value
            if model_status.torch_jit:
                key += f"-{model_status.torch_jit.value}"
            if model_status.precision:
                key += f"-{model_status.precision.value}"
            if model_status.performance is not None:
                results[key] = model_status.performance.latency
            else:
                results[key] = None
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
