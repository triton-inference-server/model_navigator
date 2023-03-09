# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""PackageDescriptor module - structure to handle execution status."""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import yaml
from packaging import version

from model_navigator.api.config import CUSTOM_CONFIGS_MAPPING, SERIALIZED_FORMATS, DeviceKind, Format, ProfilerConfig
from model_navigator.commands.base import CommandOutput, CommandStatus, ExecutionUnit
from model_navigator.commands.correctness.correctness import Correctness
from model_navigator.commands.performance.performance import Performance
from model_navigator.commands.verification.verify import VerifyModel
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.core.status import ModelStatus, RunnerStatus, Status
from model_navigator.exceptions import (
    ModelNavigatorMissingSourceModelError,
    ModelNavigatorNotFoundError,
    ModelNavigatorRuntimeAnalyzerError,
)
from model_navigator.logger import LOGGER
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import get_runner
from model_navigator.runtime_analyzer.analyzer import RuntimeAnalyzer
from model_navigator.runtime_analyzer.strategy import (
    MaxThroughputAndMinLatencyStrategy,
    MaxThroughputStrategy,
    MinLatencyStrategy,
    RuntimeSearchStrategy,
)
from model_navigator.utils import package as package_utils
from model_navigator.utils.common import get_default_status_filename, get_default_workspace
from model_navigator.utils.format_helpers import get_framework_export_formats, is_source_format
from model_navigator.utils.framework import Framework


class Package:
    """Class for storing pipeline execution status."""

    status_filename = get_default_status_filename()

    def __init__(self, status: Status, workspace: Optional[Path], model: Optional[object] = None):
        """Initialize object.

        Args:
            status: A navigator execution status
            workspace: A workspace path
            model: An optional model
        """
        self.status = status
        self.workspace = workspace
        self._model = model
        self._forward_kw_names = None

    @property
    def framework(self) -> Framework:
        """Framework for which package was created.

        Returns:
            Framework object for package
        """
        return Framework(self.status.config["framework"])

    @property
    def model(self) -> object:
        """Return source model.

        Returns:
            Source model.
        """
        return self._model

    def save_status_file(self) -> None:
        """Save the status.yaml."""
        self._delete_status_file()
        self._create_status_file()

    def get_model_path(self, model_key: str) -> Path:
        """Return path of the model.

        Args:
            model_key (str): Unique key of the model.

        Raises:
            ModelNavigatorNotFoundError: When model not found.

        Returns:
            Path: model path
        """
        try:
            model_config = self.status.models_status[model_key].model_config
        except KeyError:
            raise ModelNavigatorNotFoundError(f"Model {model_key} not found.")
        return self.workspace / model_config.path

    def save(
        self,
        path: Union[str, Path],
        keep_workspace: bool = True,
        override: bool = False,
        save_data: bool = True,
    ) -> None:
        """Save export results into the .nav package at given path.

        Args:
            path: A path to file where the package has to be saved
            keep_workspace: flag to remove the working directory after saving the package
            override: flag to override existing package in provided path
            save_data: disable saving samples from the dataloader
        """
        path = Path(path)
        if path.exists():
            if override:
                path.unlink()
            else:
                raise FileExistsError(path)

        if not self.workspace.exists():
            raise FileNotFoundError("Workspace has been removed. Save() no longer available.")

        if self.is_empty():
            LOGGER.warning("No successful exports, .nav package will be empty.")

        files_to_save = [self.workspace / "status.yaml", self.workspace / "navigator.log"]
        models_paths_to_save = self._get_models_paths_to_save(self.workspace)
        dirs_to_save = [model_path.parent for model_path in models_paths_to_save]
        if save_data:
            dirs_to_save.extend([self.workspace / "model_output", self.workspace / "model_input"])
        self._make_zip(path, self.workspace, dirs_to_save, files_to_save)

        if not keep_workspace:
            self._cleanup()

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        workspace: Optional[Union[str, Path]] = None,
    ) -> "Package":
        """Load package from provided path and updates to the current version.

        Args:
            path: The location of package to load
            workspace: Workspace where packages will be extracted

        Returns:
            Package.
        """

        def _filter_out_generated_files(paths: List[str]):
            generated_files_extensions = [".log", ".sh", ".py"]
            return [p for p in paths if not any([p.endswith(suffix) for suffix in generated_files_extensions])]

        def _extract_pkg_version(status_dict):
            return version.parse(status_dict.get("model_navigator_version", "0.3.0"))

        path = Path(path)
        if workspace is None:
            workspace = get_default_workspace()
        workspace = Path(workspace)

        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(Package.status_filename) as status_file:
                status_dict = yaml.safe_load(status_file)
            pkg_version = _extract_pkg_version(status_dict)
            status = Status.from_dict(status_dict)

            if workspace.exists():
                shutil.rmtree(workspace)

            package = Package(status, workspace)
            all_members = zf.namelist()
            filtered_members = _filter_out_generated_files(all_members)
            zf.extractall(workspace, members=filtered_members)

        package_utils.PackageUpdater().update(package, pkg_version)
        package.save_status_file()
        return package

    @property
    def config(self) -> CommonConfig:
        """Generate configuration from package.

        Returns:
            The configuration object
        """
        config_dict = {**self.status.config}
        config_dict["framework"] = self.framework
        config_dict["target_formats"] = self._target_formats
        config_dict["runner_names"] = tuple(config_dict["runner_names"])
        config_dict["profiler_config"] = ProfilerConfig.from_dict(config_dict.get("profiler_config", {}))
        if "batch_dim" not in config_dict:
            config_dict["batch_dim"] = None

        config_dict["custom_configs"] = self._get_custom_configs(self.status.config["custom_configs"])
        config_dict["target_device"] = DeviceKind(config_dict["target_device"])
        return CommonConfig(
            model=None,
            workspace=self.workspace,
            dataloader=[],
            **config_dict,
        )

    def load_source_model(self, model: object) -> None:
        """Load model defined in Python code.

        Args:
            model: A model object
        """
        if self._model is not None:
            LOGGER.warning("Overriding existing source model.")
        self._model = model

    def _get_best_runtime(
        self,
        strategy: Optional[RuntimeSearchStrategy] = None,
        include_source: bool = True,
    ):
        if strategy is None:
            strategy = MaxThroughputAndMinLatencyStrategy()

        formats = None
        if not include_source:
            formats = [fmt.value for fmt in SERIALIZED_FORMATS]

        runtime_result = RuntimeAnalyzer.get_runtime(self.status.models_status, strategy=strategy, formats=formats)
        return runtime_result

    def get_runner(
        self,
        strategy: Optional[RuntimeSearchStrategy] = None,
        include_source: bool = True,
    ) -> NavigatorRunner:
        """Get the runner according to the strategy.

        Args:
            strategy: Strategy for finding the best runtime. Defaults to `MaxThroughputAndMinLatencyStrategy`.
            include_source: Flag if Python based model has to be included in analysis

        Returns:
            The optimal runner for the optimized model.
        """
        runtime_result = self._get_best_runtime(strategy=strategy, include_source=include_source)

        model_config = runtime_result.model_status.model_config
        runner_status = runtime_result.runner_status

        if not (self.workspace / model_config.path).exists():
            raise ModelNavigatorNotFoundError(
                f"The best runner expects {model_config.format.value!r} "
                "model but it is not available in the loaded package."
            )

        if is_source_format(model_config.format) and self._model is None:
            raise ModelNavigatorMissingSourceModelError(
                "The best runner uses the source model but it is not available in the loaded package. "
                "Please load the source model with `package.load_source_model(model)` "
                "or exclude source model from optimal runner search "
                "with `package.get_optimal_runner(include_source=False)`."
            )

        return self._get_runner(model_config.key, runner_status.runner_name)

    def is_empty(self) -> bool:
        """Validate if package is empty - no models were produced.

        Returns:
            True if empty package, False otherwise.
        """
        for model_status in self.status.models_status.values():
            if not is_source_format(model_status.model_config.format):
                for runner_status in model_status.runners_status.values():
                    if (
                        runner_status.status.get(Correctness.__name__) == CommandStatus.OK
                        and runner_status.status.get(Performance.__name__) != CommandStatus.FAIL
                        and (self.workspace / model_status.model_config.path.parent).exists()
                    ):
                        return False
        return True

    def _get_runner(
        self,
        model_key: str,
        runner_name: str,
    ) -> NavigatorRunner:
        """Load runner.

        Args:
            model_key (str): Unique key of the model.
            runner_name (str): Name of the runner.

        Raises:
            ModelNavigatorNotFoundError when no runner found for provided constraints.

        Returns:
            NavigatorRunner object
        """
        try:
            model_config = self.status.models_status[model_key].model_config
        except KeyError:
            raise ModelNavigatorNotFoundError(f"Model {model_key} not found.")

        if is_source_format(model_config.format):
            model = self._model
        else:
            model = self.workspace / model_config.path
        return get_runner(runner_name)(
            model=model,
            input_metadata=self.status.input_metadata,
            output_metadata=self.status.output_metadata,
        )  # pytype: disable=not-instantiable

    def _create_status_file(self) -> None:
        """Create a status.yaml file for package."""
        path = self.workspace / self.status_filename
        with path.open("w") as f:
            yaml.safe_dump(self.status.to_dict(parse=True), f, sort_keys=False)

    def _delete_status_file(self):
        """Delete the status.yaml file from package."""
        path = self.workspace / self.status_filename
        if path.exists():
            path.unlink()

    def _update_status(
        self,
        execution_unit: ExecutionUnit,
        command_output: CommandOutput,
        shared_parameters: dict,
    ):

        self.status.input_metadata = shared_parameters.get("input_metadata", self.status.input_metadata)
        self.status.dataloader_trt_profile = shared_parameters.get(
            "dataloader_trt_profile", self.status.dataloader_trt_profile
        )
        self.status.output_metadata = shared_parameters.get("output_metadata", self.status.output_metadata)
        self.status.dataloader_max_batch_size = shared_parameters.get(
            "dataloader_max_batch_size", self.status.dataloader_max_batch_size
        )

        if execution_unit.model_config is not None:
            # If not models_status with given model_config then add new ModelStatus.
            if execution_unit.model_config.key not in self.status.models_status:
                self.status.models_status[execution_unit.model_config.key] = ModelStatus(
                    model_config=execution_unit.model_config,
                )

        if execution_unit.model_config is not None and execution_unit.runner_cls is None:
            model_status = self.status.models_status[execution_unit.model_config.key]
            model_status.status[execution_unit.command.__name__] = command_output.status
            if command_output.output:
                model_status.result[execution_unit.command.__name__] = command_output.output

        if execution_unit.model_config is not None and execution_unit.runner_cls is not None:
            model_status = self.status.models_status[execution_unit.model_config.key]
            if execution_unit.runner_cls.name() not in model_status.runners_status:
                model_status.runners_status[execution_unit.runner_cls.name()] = RunnerStatus(
                    runner_name=execution_unit.runner_cls.name(),
                )
            runners_status = model_status.runners_status[execution_unit.runner_cls.name()]
            runners_status.status[execution_unit.command.__name__] = command_output.status
            if command_output.output:
                runners_status.result[execution_unit.command.__name__] = command_output.output

    def _cleanup(self):
        if self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)

    @property
    def _target_formats(self):
        return tuple(Format(target_format) for target_format in self.status.config["target_formats"])

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

    def get_best_model_status(
        self,
        strategy: Optional[RuntimeSearchStrategy] = None,
        include_source: bool = True,
    ) -> ModelStatus:
        """Returns ModelStatus of best model for given strategy.

        Args:
            strategy: Strategy for finding the best runtime. Defaults to `MaxThroughputAndMinLatencyStrategy`.
            include_source: Flag if Python based model has to be included in analysis

        Returns:
            ModelStatus of best model for given strategy or None.
        """
        runtime_result = self._get_best_runtime(strategy=strategy, include_source=include_source)
        return runtime_result.model_status

    def _get_models_paths_to_save(
        self,
        package_path: Path,
    ) -> Sequence[Path]:
        models_paths_to_save = set()
        for model_status in self.status.models_status.values():
            format = model_status.model_config.format
            if format in get_framework_export_formats(self.framework):
                model_path = package_path / model_status.model_config.path
                if not model_path.exists():
                    LOGGER.warning(
                        f"Model not found for {model_path.parent.name}. Saving the reproduction scripts only."
                    )
                for runtime_results in model_status.runners_status.values():
                    runtime = runtime_results.runner_name
                    if runtime_results.status.get(VerifyModel.__name__) in (None, CommandStatus.SKIPPED):
                        LOGGER.warning(f"Unverified runtime: {runtime} for the {model_path.parent.name} model.")

                models_paths_to_save.add(model_path)

        for strategy in [MaxThroughputStrategy(), MinLatencyStrategy()]:
            try:
                best_model_status = self.get_best_model_status(include_source=False, strategy=strategy)
                best_format_path = package_path / best_model_status.model_config.path
                if best_format_path.exists():
                    models_paths_to_save.add(best_format_path)
            except ModelNavigatorRuntimeAnalyzerError:
                LOGGER.info(f"No model found with strategy: {strategy}")

        return tuple(models_paths_to_save)

    def _get_custom_configs(self, custom_configs: Dict[str, Dict]) -> Dict:
        """Build custom configs from config data.

        Args:
            custom_configs: Dictionary with custom configs data

        Returns:
            List with mapped objects
        """
        custom_configs_mapped = {}
        for class_name, fields in custom_configs.items():
            custom_config_class = CUSTOM_CONFIGS_MAPPING[class_name]
            obj = custom_config_class.from_dict(fields)  # pytype: disable=not-instantiable
            custom_configs_mapped[class_name] = obj

        return custom_configs_mapped
