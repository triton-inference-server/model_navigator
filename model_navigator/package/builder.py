# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Build package from pipeline context."""

import os
import pathlib
import uuid
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

from model_navigator.api.config import Format, TensorRTProfile
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.infer_metadata import InferInputMetadata, InferOutputMetadata
from model_navigator.commands.load import LoadMetadata
from model_navigator.commands.verification.verify import VerifyModel
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.core.constants import NAVIGATOR_PACKAGE_VERSION
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.exceptions import ModelNavigatorRuntimeAnalyzerError, ModelNavigatorRuntimeError
from model_navigator.pipelines.pipeline_context import PipelineCommands, PipelineContext
from model_navigator.runtime_analyzer.strategy import MaxThroughputStrategy, MinLatencyStrategy
from model_navigator.utils.format_helpers import FORMAT2SUFFIX, get_framework_export_formats

from .package import Package
from .status import ModelStatus, RunnerStatus, Status


class PackageBuilder:
    """Create package from execution context."""

    def create(self, config: CommonConfig, context: PipelineContext, model: Optional[object] = None) -> Package:
        """Create package object.

        Args:
            config: Common config used for execution
            context: Context of pipelines execution
            model: Model for which package is created

        Returns:
            Package object
        """
        status, result = self._get_command_status_and_result(context.commands)

        status = Status(
            uuid=str(uuid.uuid1()),
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=context.metadata.model_navigator_version,
            environment=context.metadata.environment,
            config=config.to_dict(),
            models_status=self._get_model_status(context.commands),
            input_metadata=self._get_input_metadata(context.commands),
            output_metadata=self._get_output_metadata(context.commands),
            dataloader_trt_profile=self._get_dataloader_trt_profile(context.commands),
            dataloader_max_batch_size=self._get_dataloader_max_batch_size(context.commands),
            status=status,
            result=result,
        )

        package = Package(status=status, workspace=context.workspace, model=model)
        package.save_status_file()

        return package

    def save(self, package: Package, path: pathlib.Path, override: bool = False, save_data: bool = True):
        """Save export results into the .nav package at given path.

        Args:
            package: Package to save
            path: A path to file where the package has to be saved
            override: flag to override existing package in provided path
            save_data: disable saving samples from the dataloader
        """
        path = pathlib.Path(path)
        if path.exists():
            if override:
                path.unlink()
            else:
                raise FileExistsError(path)

        if not package.workspace.exists():
            raise FileNotFoundError("Workspace has been removed. Save() no longer available.")

        if package.is_empty():
            LOGGER.warning("No successful exports, .nav package will be empty.")

        package.save_status_file()
        models_files_to_save = self._get_models_paths_to_save(package)
        reproduction_files_to_save = self._get_reproduction_paths_to_save(package)
        files_to_save = (
            [package.workspace.path / "status.yaml", package.workspace.path / "navigator.log"]
            + models_files_to_save
            + reproduction_files_to_save
        )
        dirs_to_save = []
        if save_data:
            dirs_to_save.extend([package.workspace.path / "model_output", package.workspace.path / "model_input"])

        self._make_zip(
            zip_path=path,
            workspace=package.workspace.path,
            dirs_to_save=dirs_to_save,
            files_to_save=files_to_save,
        )

    def _make_zip(self, zip_path, workspace, dirs_to_save, files_to_save) -> None:
        with zipfile.ZipFile(zip_path.as_posix(), "w") as zf:
            for dir_to_save in dirs_to_save:
                for root, _, files in os.walk(dir_to_save.as_posix()):
                    for filename in files:
                        zf.write(
                            os.path.join(root, filename), os.path.relpath(os.path.join(root, filename), workspace / ".")
                        )  # noqa: E203
            for filepath in files_to_save:
                zf.write(filepath, os.path.relpath(filepath, workspace / "."))  # noqa: E203

    def _get_onnx_external_weights_filepaths(self, package: Package, model_path: pathlib.Path) -> Set[pathlib.Path]:
        """Returns external weights paths for ONNX model."""
        return {fp for fp in model_path.parent.iterdir() if fp.is_file()} - set(
            self._get_reproduction_paths_to_save(package=package)
        )

    def _get_models_paths_to_save(self, package: Package) -> List[pathlib.Path]:
        models_paths_to_save = set()
        for model_status in package.status.models_status.values():
            format = model_status.model_config.format
            if format in get_framework_export_formats(package.framework):
                model_path = package.workspace.path / model_status.model_config.path
                if not model_path.exists():
                    LOGGER.warning(f"Model not found for {model_path.parent.name}.")
                    continue

                for runtime_results in model_status.runners_status.values():
                    runtime = runtime_results.runner_name
                    if runtime_results.status.get(VerifyModel.__name__) in (None, CommandStatus.SKIPPED):
                        LOGGER.warning(f"Unverified runtime: {runtime} for the {model_path.parent.name} model.")

                models_paths_to_save.add(model_path)

        for strategy in [MaxThroughputStrategy(), MinLatencyStrategy()]:
            try:
                best_model_status = package.get_best_model_status(include_source=False, strategy=strategy)
                best_format_path = package.workspace.path / best_model_status.model_config.path
                if best_format_path.exists():
                    models_paths_to_save.add(best_format_path)
            except ModelNavigatorRuntimeAnalyzerError:
                LOGGER.info(f"No model found with strategy: {strategy}")

        external_weights_paths = set()
        model_subpaths = set()
        for model_path in models_paths_to_save:
            if model_path.suffix == FORMAT2SUFFIX[Format.ONNX]:
                filepaths = self._get_onnx_external_weights_filepaths(package=package, model_path=model_path)
                external_weights_paths.update(filepaths)

            if model_path.is_dir():
                for subfile in model_path.rglob("*"):
                    model_subpaths.add(subfile)

        models_paths_to_save.update(external_weights_paths)
        models_paths_to_save.update(model_subpaths)

        return list(models_paths_to_save)

    def _get_reproduction_paths_to_save(self, package: Package) -> List[pathlib.Path]:
        reproduction_paths_to_save = set()
        for model_status in package.status.models_status.values():
            model_path = package.workspace.path / model_status.model_config.key
            if not model_path.exists():
                LOGGER.warning(f"Model path not found {model_path.name}.")
                continue

            for file in model_path.iterdir():
                if file.suffix in [".py", ".sh", ".log"]:
                    reproduction_paths_to_save.add(file)

        return list(reproduction_paths_to_save)

    def _get_command_status_and_result(
        self, commands: PipelineCommands
    ) -> Tuple[Dict[str, CommandStatus], Dict[str, Any]]:
        status = {}
        result = {}
        for command_name, command_output in commands.commands.items():
            status[command_name] = command_output.status
            if command_output.output:
                result[command_name] = command_output.output

        return status, result

    def _get_model_status(self, commands: PipelineCommands) -> Dict[str, ModelStatus]:
        model_commands = commands.models_commands
        model_status = {}
        for model_key, model_command in model_commands.items():
            runners_status = {}
            for runner_name, runner_command in model_command.runners_commands.items():
                status = {}
                result = {}
                for command_name, command_output in runner_command.commands.items():
                    status[command_name] = command_output.status
                    if command_output.output:
                        result[command_name] = command_output.output

                runners_status[runner_name] = RunnerStatus(
                    runner_name=runner_name,
                    status=status,
                    result=result,
                )

            status = {}
            result = {}
            for command_name, command_output in model_command.commands.items():
                status[command_name] = command_output.status
                if command_output.output:
                    result[command_name] = command_output.output

            model_status[model_key] = ModelStatus(
                model_config=model_command.model_config,
                runners_status=runners_status,
                status=status,
                result=result,
            )

        return model_status

    def _get_input_metadata(self, commands: PipelineCommands) -> TensorMetadata:
        command_output = commands.commands.get(InferInputMetadata.name)
        if command_output:
            return command_output.output["input_metadata"]

        command_output = commands.commands.get(LoadMetadata.name)
        if command_output:
            return command_output.output["input_metadata"]

        raise ModelNavigatorRuntimeError("Could not obtain `input_metadata`")

    def _get_output_metadata(self, commands: PipelineCommands) -> TensorMetadata:
        command_output = commands.commands.get(InferOutputMetadata.name)
        if command_output:
            return command_output.output["output_metadata"]

        command_output = commands.commands.get(LoadMetadata.name)
        if command_output:
            return command_output.output["output_metadata"]

        raise ModelNavigatorRuntimeError("Could not obtain `output_metadata`")

    def _get_dataloader_trt_profile(self, commands: PipelineCommands) -> TensorRTProfile:
        command_output = commands.commands.get(InferInputMetadata.name)
        if command_output:
            return command_output.output["dataloader_trt_profile"]

        command_output = commands.commands.get(LoadMetadata.name)
        if command_output:
            return command_output.output["dataloader_trt_profile"]

        raise ModelNavigatorRuntimeError("Could not obtain `dataloader_trt_profile`")

    def _get_dataloader_max_batch_size(self, commands: PipelineCommands) -> int:
        command_output = commands.commands.get(InferInputMetadata.name)
        if command_output:
            return command_output.output["dataloader_max_batch_size"]

        command_output = commands.commands.get(LoadMetadata.name)
        if command_output:
            return command_output.output["dataloader_max_batch_size"]

        raise ModelNavigatorRuntimeError("Could not obtain `dataloader_max_batch_size`")
