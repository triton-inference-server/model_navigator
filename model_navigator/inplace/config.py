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
"""Inplace Optimize configuration."""

import dataclasses
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

from model_navigator.api.config import CustomConfig, DeviceKind, Format, OptimizationProfile, VerifyFunction
from model_navigator.core.constants import DEFAULT_SAMPLE_COUNT
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runtime_analyzer.strategy import MinLatencyStrategy, RuntimeSearchStrategy

DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "model_navigator"
DEFAULT_MIN_NUM_SAMPLES = 100
DEFAULT_MAX_NUM_SAMPLES_STORED = 1


class InplaceConfig:
    """Inplace Optimize configuration."""

    def __init__(self) -> None:
        """Initialize InplaceConfig."""
        self._cache_dir: pathlib.Path = DEFAULT_CACHE_DIR
        self._min_num_samples: int = DEFAULT_MIN_NUM_SAMPLES
        self._max_num_samples_stored: int = DEFAULT_MAX_NUM_SAMPLES_STORED
        self.strategy: RuntimeSearchStrategy = MinLatencyStrategy()

    @property
    def min_num_samples(self) -> int:
        """Get the minimum number of samples to collect before optimizing."""
        return self._min_num_samples

    @min_num_samples.setter
    def min_num_samples(self, min_num_samples: int) -> None:
        """Set the minimum number of samples to collect before optimizing."""
        if min_num_samples < 1:
            raise ValueError(f"min_num_samples must be greater than 0, got {min_num_samples}")
        self._min_num_samples = min_num_samples

    @property
    def max_num_samples_stored(self) -> int:
        """Get the minimum number of samples to collect before optimizing."""
        return self._max_num_samples_stored

    @max_num_samples_stored.setter
    def max_num_samples_stored(self, max_num_samples_stored: int) -> None:
        """Set the minimum number of samples to collect before optimizing."""
        self._max_num_samples_stored = max_num_samples_stored

    @property
    def cache_dir(self) -> pathlib.Path:
        """Get the cache directory."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir: Union[str, pathlib.Path]) -> None:
        """Set the cache directory."""
        self._cache_dir = pathlib.Path(cache_dir)


@dataclasses.dataclass
class OptimizeConfig:
    """Configuration for inplace Optimize.

    Args:
        sample_count: Limits how many samples will be used from dataloader
        batching: Enable or disable batching on first (index 0) dimension of the model
        input_names: Model input names
        output_names: Model output names
        target_formats: Target model formats for optimize process
        target_device: Target device for optimize process, default is CUDA
        runners: Use only runners provided as parameter
        optimization_profile: Optimization profile for conversion and profiling
        workspace: Workspace where packages will be extracted
        verbose: Enable verbose logging
        debug: Enable debug logging from commands
        verify_func: Function for additional model verification
        custom_configs: Sequence of CustomConfigs used to control produced artifacts
    """

    sample_count: int = DEFAULT_SAMPLE_COUNT
    batching: Optional[bool] = True
    input_names: Optional[Tuple[str, ...]] = None
    output_names: Optional[Tuple[str, ...]] = None
    target_formats: Optional[Tuple[Union[str, Format], ...]] = None
    target_device: Optional[DeviceKind] = DeviceKind.CUDA
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None
    optimization_profile: Optional[OptimizationProfile] = None
    workspace: Optional[pathlib.Path] = None
    verbose: Optional[bool] = False
    debug: Optional[bool] = False
    verify_func: Optional[VerifyFunction] = None
    custom_configs: Optional[Sequence[CustomConfig]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert OptimizeConfig to dictionary."""
        config_dict = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            config_dict[field.name] = value
        return config_dict


inplace_config = InplaceConfig()
