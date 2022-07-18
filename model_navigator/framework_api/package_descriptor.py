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
from importlib.metadata import version
from itertools import chain
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import yaml

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.exceptions import UserError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.pipelines.builders.profiling import profiling_builder
from model_navigator.framework_api.pipelines.pipeline_manager import PipelineManager
from model_navigator.framework_api.status import ModelStatus, NavigatorStatus, RuntimeResults
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

NAV_PACKAGE_FORMAT_VERSION = "0.1.2"


class PackageDescriptor:
    status_filename = get_default_status_filename()

    def __init__(self, navigator_status: NavigatorStatus, workdir: Path, model: Optional[object] = None):
        self.navigator_status = navigator_status
        self.workdir = workdir
        self.model = model

    @classmethod
    def build(cls, pipeline_manager: PipelineManager, config: Config):
        config_filter_fields = [
            "model",
            "dataloader",
            "workdir",
            "override_workdir",
            "forward_kw_names",
            "disable_git_info",
            "input_metadata",
            "output_metadata",
        ]
        navigator_status = NavigatorStatus(
            uuid=str(uuid.uuid1()),
            format_version=NAV_PACKAGE_FORMAT_VERSION,
            model_navigator_version=version("model_navigator"),
            git_info=get_git_info(config.disable_git_info),
            environment=get_env(),
            export_config=config.to_dict(
                config_filter_fields,
                parse=True,
            ),
            model_status=[],
            input_metadata=TensorMetadata(),
            output_metadata=TensorMetadata(),
        )

        pkg_desc = cls(navigator_status, config.workdir, model=config.model)
        pipeline_manager.run(config=config, package_descriptor=pkg_desc)
        pkg_desc.navigator_status.export_config = config.to_dict(
            config_filter_fields,
            parse=True,
        )
        pkg_desc.save_status_file()

        LOGGER.warning(
            "Initially models are not verified. Validate exported models and use "
            "PackageDescriptor.set_verified(format, runtime, jit_type, precision) method to set models as verified."
        )

        return pkg_desc

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

    def save_status_file(self):
        self.delete_status_file()
        self.create_status_file()

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
        LOGGER.debug(f"Loading runner from path: {model_path}")

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
            from model_navigator.framework_api.runners.pyt import PytRunner

            return PytRunner(
                model_path,
                input_metadata=self.navigator_status.input_metadata,
                output_names=list(self.navigator_status.output_metadata.keys()),
                target_device=self.navigator_status.export_config["target_device"],
            )
        elif format == Format.TF_SAVEDMODEL:
            from model_navigator.framework_api.runners.tf import TFRunner

            return TFRunner(
                model_path,
                input_metadata=self.navigator_status.input_metadata,
                output_names=list(self.navigator_status.output_metadata.keys()),
            )
        elif format == Format.TF_TRT:
            from model_navigator.framework_api.runners.tf import TFRunner

            return TFRunner(
                model_path,
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

    def _get_model_status_name(self, model_status: ModelStatus) -> str:
        name = model_status.format.value
        if model_status.torch_jit:
            name += f"-{model_status.torch_jit.value}"
        if model_status.precision:
            name += f"-{model_status.precision.value}"
        return name

    def get_formats_status(self) -> Dict:
        """Return dictionary of pairs Format : Bool. True for successful exports, False for failed exports."""
        results = {}
        for model_status in self.navigator_status.model_status:
            key = self._get_model_status_name(model_status)
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
    ) -> ModelStatus:
        for model_status in self.navigator_status.model_status:
            if (
                model_status.format == format
                and (model_status.torch_jit in (None, jit_type))
                and (model_status.precision in (None, precision))
            ):
                return model_status
        raise RuntimeError(f"Model status not found for {format=}, {jit_type=}, {precision=}.")

    def get_runtime_results(
        self,
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
        runtime_provider: Optional[RuntimeProvider] = None,
    ) -> RuntimeResults:
        model_status = self._get_model_status(format, jit_type, precision)
        if not runtime_provider and len(model_status.runtime_results) == 1:
            return model_status.runtime_results[0]
        for runtime_results in model_status.runtime_results:
            if runtime_results.runtime == runtime_provider:
                return runtime_results
        raise RuntimeError(f"Model status not found for {format=}, {jit_type=}, {precision=}, {runtime_provider=}.")

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
            prec_iter = (
                [None] if format not in (Format.TENSORRT, Format.TF_TRT, Format.TORCH_TRT) else self._target_precisions
            )
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
    ) -> "PackageDescriptor":
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

    @property
    def config(self):
        config_dict = {**self.navigator_status.export_config}
        config_dict["framework"] = self.framework
        config_dict["target_formats"] = self._target_formats
        config_dict["target_jit_type"] = self._target_jit_type
        config_dict["target_precisions"] = self._target_precisions
        config_dict["onnx_runtimes"] = tuple(RuntimeProvider(prov) for prov in config_dict["onnx_runtimes"])
        config_dict["profiler_config"] = ProfilerConfig.from_dict(config_dict.get("profiler_config", {}))
        if "batch_dim" not in config_dict:
            config_dict["batch_dim"] = None
        return Config(
            model=None,
            workdir=self.workdir,
            dataloader=[],
            override_workdir=False,
            disable_git_info=True,
            **config_dict,
        )

    def profile(self, profiler_config: Optional[ProfilerConfig] = None) -> None:
        """
        Run profiling on the package for each batch size from the `batch_sizes`.
        """
        config = self.config
        config.profiler_config = profiler_config

        pipeline_manager = PipelineManager([profiling_builder])
        pipeline_manager.run(config=config, package_descriptor=self)
        self.save_status_file()


def save(
    package_descriptor: PackageDescriptor,
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
    package_descriptor.save(
        path=path,
        keep_workdir=keep_workdir,
        override=override,
        save_data=save_data,
    )


def profile(package_descriptor: PackageDescriptor, profiler_config: Optional[ProfilerConfig] = None) -> None:
    """
    Run profiling on the package for each batch size from the `profiler_config.batch_sizes`.
    """
    package_descriptor.profile(profiler_config=profiler_config)
