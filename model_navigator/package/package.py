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
"""Package module - structure to snapshot optimization result."""

import copy
import pathlib
from typing import Dict, Optional, Union

import yaml

from model_navigator.api.config import (
    CUSTOM_CONFIGS_MAPPING,
    SERIALIZED_FORMATS,
    CustomConfigForFormat,
    DeviceKind,
    Format,
    OptimizationProfile,
    TensorType,
)
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.correctness.correctness import Correctness
from model_navigator.commands.performance.performance import Performance
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.core.logger import LOGGER
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorMissingSourceModelError, ModelNavigatorNotFoundError
from model_navigator.frameworks import Framework
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.registry import get_runner
from model_navigator.runtime_analyzer.analyzer import RuntimeAnalyzer
from model_navigator.runtime_analyzer.strategy import MaxThroughputAndMinLatencyStrategy, RuntimeSearchStrategy
from model_navigator.utils.common import DataObject, get_default_status_filename
from model_navigator.utils.format_helpers import is_source_format

from .status import ModelStatus, Status


class Package:
    """Class for storing pipeline execution status."""

    status_filename = get_default_status_filename()

    def __init__(self, status: Status, workspace: Workspace, model: Optional[object] = None):
        """Initialize object.

        Args:
            status: A navigator execution status
            workspace: Workspace for package files
            model: An optional model
        """
        self.status = status
        self.workspace = workspace
        self._model = model

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

        optimization_profile = config_dict.get("optimization_profile", {})
        if isinstance(optimization_profile, dict):
            optimization_profile = OptimizationProfile.from_dict(optimization_profile)
        config_dict["optimization_profile"] = optimization_profile

        if "batch_dim" not in config_dict:
            config_dict["batch_dim"] = None

        config_dict["custom_configs"] = self._get_custom_configs(
            self.status.config["custom_configs"]
        )  # pytype: disable=wrong-arg-types
        config_dict["target_device"] = DeviceKind(config_dict["target_device"])

        if "model" not in config_dict:
            config_dict["model"] = self.model

        if "dataloader" not in config_dict:
            config_dict["dataloader"] = []

        return CommonConfig(
            **config_dict,
        )

    def save_status_file(self) -> None:
        """Save the status.yaml."""
        self._delete_status_file()
        self._create_status_file()

    def get_model_path(self, model_key: str) -> pathlib.Path:
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
            raise ModelNavigatorNotFoundError(f"Model {model_key} not found.") from None
        return self.workspace.path / model_config.path

    def load_source_model(self, model: object) -> None:
        """Load model defined in Python code.

        Args:
            model: A model object
        """
        if self._model is not None:
            LOGGER.warning("Overriding existing source model.")
        self._model = model

    def get_runner(
        self,
        strategy: Optional[RuntimeSearchStrategy] = None,
        include_source: bool = True,
        return_type: TensorType = TensorType.NUMPY,
    ) -> NavigatorRunner:
        """Get the runner according to the strategy.

        Args:
            strategy: Strategy for finding the best runtime. Defaults to `MaxThroughputAndMinLatencyStrategy`.
            include_source: Flag if Python based model has to be included in analysis
            return_type: The type of the output tensor. Defaults to `TensorType.NUMPY`.
                If the return_type supports CUDA tensors (e.g. TensorType.TORCH) and the input tensors are on CUDA,
                there will be no additional data transfer between CPU and GPU.

        Returns:
            The optimal runner for the optimized model.
        """
        runtime_result = self._get_best_runtime(strategy=strategy, include_source=include_source)

        model_config = runtime_result.model_status.model_config
        runner_status = runtime_result.runner_status

        if not is_source_format(model_config.format) and not (self.workspace.path / model_config.path).exists():
            raise ModelNavigatorNotFoundError(
                f"The best runner expects {model_config.format.value!r} "
                "model but it is not available in the loaded package."
            )

        if is_source_format(model_config.format) and self._model is None:
            raise ModelNavigatorMissingSourceModelError(
                "The best runner uses the source model but it is not available in the loaded package. "
                "Please load the source model with `package.load_source_model(model)` "
                "or exclude source model from optimal runner search "
                "with `package.get_runner(include_source=False)`."
            )

        return self._get_runner(model_config.key, runner_status.runner_name, return_type=return_type)

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
                        and (self.workspace.path / model_status.model_config.path.parent).exists()
                    ):
                        return False
        return True

    def _get_runner(
        self,
        model_key: str,
        runner_name: str,
        return_type: TensorType = TensorType.NUMPY,
    ) -> NavigatorRunner:
        """Load runner.

        Args:
            model_key (str): Unique key of the model.
            runner_name (str): Name of the runner.
            return_type (TensorType): Type of the runner output.

        Raises:
            ModelNavigatorNotFoundError when no runner found for provided constraints.

        Returns:
            NavigatorRunner object
        """
        try:
            model_config = self.status.models_status[model_key].model_config
        except KeyError:
            raise ModelNavigatorNotFoundError(f"Model {model_key} not found.") from None

        if is_source_format(model_config.format):
            model = self._model
        else:
            model = self.workspace.path / model_config.path
        return get_runner(runner_name)(
            model=model,
            input_metadata=self.status.input_metadata,
            output_metadata=self.status.output_metadata,
            return_type=return_type,
        )  # pytype: disable=not-instantiable

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

    def _status_serializable_dict(self) -> Dict:
        """Convert status to serializable dict."""
        config = DataObject.filter_data(
            data=self.status.config,
            filter_fields=[
                "model",
                "dataloader",
                "verify_func",
                "workspace",
            ],
        )
        config = DataObject.parse_data(config)
        status = copy.copy(self.status)
        status.config = config
        data = status.to_dict(parse=True)
        return data

    def _create_status_file(self) -> None:
        """Create a status.yaml file for package."""
        path = self.workspace.path / self.status_filename
        data = self._status_serializable_dict()
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _delete_status_file(self):
        """Delete the status.yaml file from package."""
        path = self.workspace.path / self.status_filename
        if path.exists():
            path.unlink()

    @property
    def _target_formats(self):
        return tuple(Format(target_format) for target_format in self.status.config["target_formats"])

    def _get_custom_configs(self, custom_configs: Dict[str, Union[Dict, CustomConfigForFormat]]) -> Dict:
        """Build custom configs from config data.

        Args:
            custom_configs: Dictionary with custom configs data

        Returns:
            List with mapped objects
        """
        custom_configs_mapped = {}
        for class_name, obj in custom_configs.items():
            if isinstance(obj, dict):
                custom_config_class = CUSTOM_CONFIGS_MAPPING[class_name]
                obj = custom_config_class.from_dict(obj)  # pytype: disable=not-instantiable

            custom_configs_mapped[class_name] = obj

        return custom_configs_mapped
