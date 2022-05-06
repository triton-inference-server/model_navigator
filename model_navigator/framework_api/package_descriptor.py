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


import itertools
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.correctness.base import TolerancePerOutputName
from model_navigator.framework_api.commands.performance.base import Performance
from model_navigator.framework_api.common import DataObject, TensorMetadata
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.exceptions import UserError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.utils import (
    Extension,
    Framework,
    JitType,
    RuntimeProvider,
    Status,
    format2runtimes,
    format_to_relative_model_path,
    get_base_format,
    get_default_status_filename,
    get_default_workdir,
    get_framework_export_formats,
    get_package_path,
)
from model_navigator.model import Format
from model_navigator.utils.environment import get_env, get_git_info

NAV_PACKAGE_FORMAT_VERSION = "0.1.0"


@dataclass
class RuntimeResults(DataObject):
    runtime: RuntimeProvider
    status: Status
    tolerance: Optional[TolerancePerOutputName] = None
    performance: Optional[List[Performance]] = None
    err_msg: Optional[dict] = None
    verified: bool = False

    @classmethod
    def from_dict(cls, dict: Mapping):
        return cls(
            runtime=RuntimeProvider(dict["runtime"]),
            status=Status(dict["status"]),
            tolerance=dict.get("tolerance"),
            performance=[Performance.from_dict(perf) for perf in dict.get("performance", [])],
            err_msg=dict.get("err_msg"),
            verified=dict["verified"],
        )


@dataclass
class ModelStatus(DataObject):
    format: Format
    path: Path
    runtime_results: List[RuntimeResults]
    torch_jit: Optional[JitType] = None
    precision: Optional[TensorRTPrecision] = None

    @classmethod
    def from_dict(cls, dict: Mapping):
        return cls(
            format=Format(dict["format"]),
            path=Path(dict["path"]),
            runtime_results=[RuntimeResults.from_dict(runtime_results) for runtime_results in dict["runtime_results"]],
            torch_jit=JitType(dict["torch_jit"]) if "torch_jit" in dict else None,
            precision=TensorRTPrecision(dict["precision"]) if "precision" in dict else None,
        )


@dataclass
class NavigatorStatus(DataObject):
    format_version: str
    uuid: str
    git_info: Dict
    environment: Dict
    export_config: Dict
    model_status: List[ModelStatus]
    input_metadata: TensorMetadata
    output_metadata: TensorMetadata

    @classmethod
    def from_dict(cls, dict: Mapping):
        return cls(
            format_version=dict["format_version"],
            uuid=dict["uuid"],
            git_info=dict["git_info"],
            environment=dict["environment"],
            export_config=dict["export_config"],
            model_status=[ModelStatus.from_dict(model_status) for model_status in dict["model_status"]],
            input_metadata=TensorMetadata.from_json(dict["input_metadata"]),
            output_metadata=TensorMetadata.from_json(dict["output_metadata"]),
        )


class PackageDescriptor:
    status_filename = get_default_status_filename()

    def __init__(self, navigator_status: NavigatorStatus, workdir: Path, model: Optional[object] = None):
        self.navigator_status = navigator_status
        self.workdir = workdir
        self.model = model

    @classmethod
    def from_pipelines(cls, pipelines: List[Pipeline], config: Config):
        model_status = []
        for pipeline in pipelines:
            for command in pipeline.commands:
                if command.command_type in (CommandType.EXPORT, CommandType.CONVERT, CommandType.COPY):
                    runtime_results = []
                    runtimes = (
                        config.onnx_runtimes
                        if command.target_format == Format.ONNX
                        else format2runtimes(command.target_format)
                    )
                    for runtime_provider in runtimes:
                        correctness_results = cls._get_correctness_command_for_model(
                            commands=pipeline.commands,
                            format=command.target_format,
                            jit_type=command.target_jit_type,
                            precision=command.target_precision,
                            runtime_provider=runtime_provider,
                        )
                        performance_results = cls._get_performance_command_for_model(
                            commands=pipeline.commands,
                            format=command.target_format,
                            jit_type=command.target_jit_type,
                            precision=command.target_precision,
                            runtime_provider=runtime_provider,
                        )

                        per_output_tolerance = None
                        # Status.OK because for ONNX input because there is no Correctness.
                        status = Status.OK
                        err_msg = None
                        if correctness_results:
                            if correctness_results.output:
                                per_output_tolerance = correctness_results.output

                            status = correctness_results.status
                            err_msg = cls.get_err_msg(command, correctness_results, performance_results)

                        if (
                            performance_results
                            and performance_results.output
                            and performance_results.status == Status.OK
                        ):
                            perf = performance_results.output
                        else:
                            perf = None

                        runtime_results.append(
                            RuntimeResults(
                                runtime=runtime_provider,
                                status=status,
                                tolerance=per_output_tolerance,
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

        if not config.disable_git_info:
            git_info = get_git_info()
        else:
            git_info = None
        filtered_config = config.to_dict(
            filter_fields=[
                "model",
                "dataloader",
                "workdir",
                "override_workdir",
                "forward_kw_names",
                "disable_git_info",
                "input_metadata",
                "output_metadata",
            ],
            parse=True,
        )
        navigator_status = NavigatorStatus(
            uuid=str(uuid.uuid1()),
            format_version=NAV_PACKAGE_FORMAT_VERSION,
            git_info=git_info,
            environment=get_env(),
            export_config=filtered_config,
            model_status=model_status,
            input_metadata=config.input_metadata,
            output_metadata=config.output_metadata,
        )

        pkg_desc = cls(navigator_status, config.workdir, model=config.model)
        pkg_desc.delete_status_file()
        pkg_desc.create_status_file()

        LOGGER.warning(
            "Initially models are not verified. Validate exported models and use "
            "PackageDescriptor.set_verified(format, runtime, jit_type, precision) method to set models as verified."
        )

        return pkg_desc

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

    @property
    def model_name(self):
        return self.navigator_status.export_config["model_name"]

    @property
    def framework(self):
        return Framework(self.navigator_status.export_config["framework"])

    def create_status_file(self):
        path = get_package_path(self.workdir, self.model_name) / self.status_filename
        with open(path, "w") as f:
            yaml.safe_dump(self.navigator_status.to_dict(parse=True), f, sort_keys=False)

    def delete_status_file(self):
        path = get_package_path(self.workdir, self.model_name) / self.status_filename
        if path.exists():
            path.unlink()

    @staticmethod
    def _load_model(model_path: Path, format: Format):
        model_path = model_path.as_posix()
        LOGGER.info(f"Loading model from path: {model_path}")

        if format == Format.ONNX:
            import onnx

            return onnx.load_model(model_path)
        elif format == Format.TENSORRT:
            return model_path
        elif format in (Format.TORCHSCRIPT, Format.TORCH_TRT):
            import torch  # pytype: disable=import-error

            return torch.jit.load(model_path)
        else:
            import tensorflow  # pytype: disable=import-error

            return tensorflow.keras.models.load_model(model_path)

    def _load_runner(self, model_path: Path, format: Format, runtime: Optional[RuntimeProvider] = None):
        model_path = model_path.as_posix()
        LOGGER.info(f"Loading runner from path: {model_path}")

        if runtime is None:
            runtime = format2runtimes(format)

        if format == Format.ONNX:
            from polygraphy.backend.onnxrt import SessionFromOnnx

            from model_navigator.framework_api.runners.onnx import OnnxrtRunner

            if not isinstance(runtime, (tuple, list)):
                runtime = [runtime]
            return OnnxrtRunner(SessionFromOnnx(model_path, providers=runtime))
        elif format == Format.TENSORRT:
            from polygraphy.backend.common import BytesFromPath
            from polygraphy.backend.trt import EngineFromBytes

            from model_navigator.framework_api.runners.trt import TrtRunner

            return TrtRunner(EngineFromBytes(BytesFromPath(model_path)))
        elif format in (Format.TORCHSCRIPT, Format.TORCH_TRT):
            import torch  # pytype: disable=import-error

            from model_navigator.framework_api.runners.pyt import PytRunner

            return PytRunner(
                torch.jit.load(model_path),
                input_metadata=self.navigator_status.input_metadata,
                output_names=list(self.navigator_status.output_metadata.keys()),
                target_device=self.navigator_status.export_config["target_device"],
            )
        elif format == Format.TF_SAVEDMODEL:
            import tensorflow  # pytype: disable=import-error

            from model_navigator.framework_api.runners.tf import TFRunner

            return TFRunner(
                tensorflow.keras.models.load_model(model_path),
                input_metadata=self.navigator_status.input_metadata,
                output_names=list(self.navigator_status.output_metadata.keys()),
            )
        elif format == Format.TF_TRT:
            import tensorflow  # pytype: disable=import-error

            from model_navigator.framework_api.runners.tf import TFTRTRunner

            return TFTRTRunner(
                tensorflow.keras.models.load_model(model_path),
                input_metadata=self.navigator_status.input_metadata,
                output_names=list(self.navigator_status.output_metadata.keys()),
            )
        else:
            raise ValueError(f"Unknown format: {format}")

    def _cleanup(self):
        if self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)

    def get_verified_status(
        self,
        format: Format,
        runtime: RuntimeProvider,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
    ):
        for model_status in self.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                if (
                    model_status.format == format
                    and model_status.torch_jit == jit_type
                    and model_status.precision == precision
                    and runtime_results.runtime == runtime
                ):
                    return runtime_results.verified

    def set_verified(
        self,
        format: Format,
        runtime: RuntimeProvider,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
    ):
        """Set exported model verified for given format, jit_type and precision"""
        for model_status in self.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                if (
                    model_status.format == format
                    and model_status.torch_jit == jit_type
                    and model_status.precision == precision
                    and runtime_results.runtime == runtime
                ):
                    runtime_results.verified = True
                    self.delete_status_file()
                    self.create_status_file()
                    return
        raise UserError("Runtime not found.")

    def get_status(
        self,
        format: Format,
        runtime_provider: Optional[RuntimeProvider] = None,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
    ) -> bool:
        """Return status (True or False) of export operation for particular format, jit_type,
        precision and runtime_provider."""
        if runtime_provider is None:
            runtime_provider = format2runtimes(format)[0]

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
            key = model_status.format.value
            if model_status.torch_jit:
                key += f"-{model_status.torch_jit.value}"
            if model_status.precision:
                key += f"-{model_status.precision.value}"
            results[key] = {}
            for runtime_results in model_status.runtime_results:
                results[key][runtime_results.runtime.value] = runtime_results.status
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
        """
        Load exported model for given format, jit_type and precision and return model object

        :return
            model object for TensorFlow, PyTorch and ONNX
            model path for TensorRT
        """
        model_path = get_package_path(workdir=self.workdir, model_name=self.model_name) / format_to_relative_model_path(
            format=format, jit_type=jit_type, precision=precision
        )
        if model_path.exists():
            return self._load_model(model_path=model_path, format=format)
        else:
            return None

    def get_runner(
        self,
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
        runtime: Optional[RuntimeProvider] = None,
    ):
        """
        Load exported model for given format, jit_type and precision and return Polygraphy runner for given runtime.

        :return
            Polygraphy BaseRunner object: https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/backend/base/runner.py
        """
        model_path = get_package_path(workdir=self.workdir, model_name=self.model_name) / format_to_relative_model_path(
            format=format, jit_type=jit_type, precision=precision
        )
        if model_path.exists():
            return self._load_runner(model_path=model_path, format=format, runtime=runtime)
        else:
            return None

    def get_source_runner(
        self,
    ):
        """
        Load Polygraphy runner for source model.

        :return
            Polygraphy BaseRunner object: https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/backend/base/runner.py
        """
        if self.model is None:
            LOGGER.warning("Source model not available.")
            return None
        if self.framework == Framework.PYT:
            from model_navigator.framework_api.runners.pyt import PytRunner

            return PytRunner(
                self.model,
                input_metadata=self.navigator_status.input_metadata,
                output_names=list(self.navigator_status.output_metadata.keys()),
                target_device=self.navigator_status.export_config["target_device"],
            )
        elif self.framework == Framework.TF2:
            from model_navigator.framework_api.runners.tf import TFRunner

            return TFRunner(
                self.model,
                input_metadata=self.navigator_status.input_metadata,
                output_names=list(self.navigator_status.output_metadata.keys()),
            )
        elif self.framework == Framework.ONNX:
            from polygraphy.backend.onnxrt import SessionFromOnnx

            from model_navigator.framework_api.runners.onnx import OnnxrtRunner

            return OnnxrtRunner(SessionFromOnnx(self.model, providers=format2runtimes(Format.ONNX)))
        else:
            raise RuntimeError(f"Unknown framework: {self.framework}.")

    @property
    def _target_formats(self):
        return tuple(Format(target_format) for target_format in self.navigator_status.export_config["target_formats"])

    @property
    def _target_jit_type(self):
        return tuple(JitType(jit_type) for jit_type in self.navigator_status.export_config.get("target_jit_type", []))

    @property
    def _target_precisions(self):
        return tuple(
            TensorRTPrecision(prec) for prec in self.navigator_status.export_config.get("target_precisions", [])
        )

    def _get_base_formats(self, target_formats: Sequence[Format]) -> Tuple[Format]:
        base_formats = set()
        for target_format in target_formats:
            base_format = get_base_format(target_format, self.framework)
            if base_format is not None:
                base_formats.add(base_format)
        return tuple(base_formats)

    def _get_model_status(
        self,
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
    ):
        for model_status in self.navigator_status.model_status:
            if (
                model_status.format == format
                and (model_status.torch_jit in (None, jit_type))
                and (model_status.precision in (None, precision))
            ):
                return model_status
        raise RuntimeError(f"Model status not found for {format=}, {jit_type=}, {precision=}.")

    def _make_zip(self, zip_path, package_path, dirs_to_save, base_models_paths) -> None:

        checkpoint_extensions = {ext.value for ext in Extension}
        with zipfile.ZipFile(zip_path.as_posix(), "w") as zf:
            for dirname, _, files in os.walk(package_path.as_posix()):
                if dirname != package_path.as_posix() and not dirname.startswith(dirs_to_save):
                    continue
                for filename in files:
                    filepath = os.path.join(dirname, filename)
                    _, ext = os.path.splitext(filepath)
                    if ext.lstrip(".") in checkpoint_extensions and filepath not in [
                        mp.as_posix() for mp in base_models_paths
                    ]:
                        continue
                    zf.write(filepath, filepath[len(package_path.as_posix()) :])

    def _get_models_paths_to_save(
        self,
        package_path: Path,
    ) -> Tuple[Sequence[Path], Sequence[Path]]:
        base_formats = self._get_base_formats(self._target_formats)
        base_models_paths, converted_models_paths = set(), set()
        for format in chain(self._target_formats, base_formats):
            jit_iter = [None] if format not in (Format.TORCHSCRIPT, Format.TORCH_TRT) else self._target_jit_type
            prec_iter = [None] if format not in (Format.TENSORRT, Format.TF_TRT) else self._target_precisions
            for jit_type, prec in itertools.product(jit_iter, prec_iter):
                model_path = package_path / format_to_relative_model_path(format, jit_type, prec)
                if not model_path.exists():
                    LOGGER.warning(f"Model not found for {model_path.parent.name}.")
                    continue
                model_status = self._get_model_status(format, jit_type, prec)
                if format in self._target_formats:
                    for runtime_status in model_status.runtime_results:
                        runtime = runtime_status.runtime
                        verified_status = self.get_verified_status(
                            format=format, runtime=runtime, jit_type=jit_type, precision=prec
                        )
                        if not verified_status:
                            LOGGER.warning(f"Unverified runtime: {runtime} for the {model_path.parent.name} model.")
                        if format in get_framework_export_formats(self.framework):
                            base_models_paths.add(model_path)
                        else:
                            converted_models_paths.add(model_path)
                if format in base_formats:
                    base_models_paths.add(model_path)
        return tuple(base_models_paths), tuple(converted_models_paths)

    @property
    def _is_empty(self):
        for model_status in self.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                if runtime_results.status == Status.OK:
                    return False
        return True

    def save(
        self,
        path: Union[str, Path],
        keep_workdir: bool = True,
        override: bool = False,
        save_data: bool = True,
    ) -> None:
        """Save export results into the .nav package at given path.
        If `keep_workdir = False` remove the working directory.
        If `override = True` override `path`.
        If `save_data = False` discard samples from the dataloader.
        That won't allow for correctness check later on in the deployment process.
        """
        path = Path(path)
        if path.exists():
            if override:
                path.unlink()
            else:
                raise FileExistsError(path)

        package_path = get_package_path(workdir=self.workdir, model_name=self.model_name)

        if not package_path.exists():
            raise FileNotFoundError("Workdir has been removed. Save() no longer available.")

        if self._is_empty:
            raise RuntimeError("No successful exports, .nav package cannot be created.")

        base_models_paths, converted_models_paths = self._get_models_paths_to_save(
            package_path,
        )

        dirs_to_save = [model_path.parent for model_path in chain(base_models_paths, converted_models_paths)]
        if save_data:
            dirs_to_save.extend([package_path / "model_output", package_path / "model_input"])
        dirs_to_save = tuple(dirname.as_posix() for dirname in dirs_to_save)

        self._make_zip(path, package_path, dirs_to_save, base_models_paths)

        if not keep_workdir:
            self._cleanup()

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        workdir: Optional[Union[str, Path]] = None,
        override_workdir: bool = False,
    ):
        path = Path(path)
        if workdir is None:
            workdir = get_default_workdir()
        workdir = Path(workdir)

        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(cls.status_filename) as status_file:
                status_dict = yaml.safe_load(status_file)
            navigator_status = NavigatorStatus.from_dict(status_dict)

            package_path = get_package_path(workdir=workdir, model_name=navigator_status.export_config["model_name"])
            if package_path.exists():
                if override_workdir:
                    shutil.rmtree(package_path)
                else:
                    raise FileExistsError(package_path)
            zf.extractall(package_path)

        return cls(navigator_status, workdir)
