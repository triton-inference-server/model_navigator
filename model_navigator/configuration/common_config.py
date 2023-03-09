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
"""Config object to handle user inputs and define the execution of commands."""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from model_navigator.api.config import CustomConfig, DeviceKind, Format, ProfilerConfig, SizedDataLoader, VerifyFunction
from model_navigator.logger import LOGGER
from model_navigator.utils.common import DataObject, pad_string
from model_navigator.utils.framework import Framework


@dataclass
class CommonConfig(DataObject):
    """Command conxtext stores paramters used during commands execution not related to any particular model format."""

    framework: Framework
    model: object
    dataloader: SizedDataLoader
    workspace: Path
    target_formats: Tuple[Format, ...]
    target_device: DeviceKind
    sample_count: int
    profiler_config: ProfilerConfig
    runner_names: Tuple[str, ...]
    batch_dim: Optional[int] = 0
    seed: int = 0
    _input_names: Optional[Tuple[str, ...]] = None
    _output_names: Optional[Tuple[str, ...]] = None
    from_source: bool = True
    forward_kw_names: Optional[Tuple[str, ...]] = None
    verify_func: Optional[VerifyFunction] = None
    custom_configs: Dict[str, CustomConfig] = dataclasses.field(default_factory=lambda: {})

    # Verbose logging - enable debug mode in export and conversion paths
    verbose: bool = False

    # Debug - enabled debug mode for converters
    debug: bool = False

    def log(self) -> None:
        """Display the configuration as formatted string."""
        LOGGER.info(pad_string("Common config parameters"))
        log_dict = self.to_dict(
            filter_fields=[
                "model",
                "dataloader",
                "verify_func",
            ],
            parse=True,
        )
        LOGGER.info(json.dumps(log_dict, indent=4))
