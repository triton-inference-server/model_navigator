# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Status definition for results of inplace optimize and profile."""

import copy
import dataclasses
import pathlib
from datetime import datetime, timezone
from typing import Dict, Union

import yaml

from model_navigator.commands.base import CommandStatus
from model_navigator.configuration import TensorRTProfile
from model_navigator.core.tensor import TensorMetadata
from model_navigator.inplace.profiling import ProfilingResults
from model_navigator.package.status import ModelStatus, Status
from model_navigator.utils.common import DataObject


@dataclasses.dataclass
class ModuleStatus(DataObject):
    """Inplace Module Status."""

    format_version: str
    uuid: str
    config: Dict
    models_status: Dict[str, ModelStatus]
    input_metadata: TensorMetadata
    output_metadata: TensorMetadata
    dataloader_trt_profile: TensorRTProfile
    dataloader_max_batch_size: int
    status: Dict[str, CommandStatus]
    result: Dict
    timestamp: str

    @classmethod
    def from_package_status(cls, status: Status) -> "ModuleStatus":
        """Create Module Status object based on Navigator Package status.

        Args:
            status: Package Status object

        Returns:
            Module Status object
        """
        return ModuleStatus(
            uuid=status.uuid,
            format_version=status.format_version,
            config=status.config,
            models_status=status.models_status,
            input_metadata=status.input_metadata,
            output_metadata=status.output_metadata,
            dataloader_trt_profile=status.dataloader_trt_profile,
            dataloader_max_batch_size=status.dataloader_max_batch_size,
            status=status.status,
            result=status.result,
            timestamp=status.timestamp,
        )


@dataclasses.dataclass
class InplaceOptimizeStatus(DataObject):
    """Inplace Optimize Status."""

    status_version: str
    model_navigator_version: str
    uuid: str
    environment: Dict
    module_status: Dict[str, ModuleStatus]
    timestamp: str = dataclasses.field(default_factory=lambda: f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%S.%f}")

    def to_file(self, path: Union[pathlib.Path, str]):
        """Save the status to file.

        Args:
            path: Path where status should be saved
        """
        path = pathlib.Path(path)
        data = self._status_serializable_dict()
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _status_serializable_dict(self) -> Dict:
        """Convert status to serializable dict."""
        status = copy.copy(self)
        for ms in status.module_status.values():
            config = DataObject.filter_data(
                data=ms.config,
                filter_fields=[
                    "model",
                    "dataloader",
                    "verify_func",
                    "workspace",
                ],
            )
            config = DataObject.parse_data(config)
            ms.config = config

        data = status.to_dict(parse=True)
        return data


@dataclasses.dataclass
class InplaceProfileStatus(DataObject):
    """Inplace Profile Status."""

    status_version: str
    model_navigator_version: str
    uuid: str
    environment: Dict
    profiling_results: ProfilingResults
    timestamp: str = dataclasses.field(default_factory=lambda: f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%S.%f}")

    def to_file(self, path: Union[pathlib.Path, str]):
        """Save the status to file.

        Args:
            path: Path where status should be saved
        """
        data = self.to_dict(parse=True)

        path = pathlib.Path(path)
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
