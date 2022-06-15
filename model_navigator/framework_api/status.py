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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.correctness import TolerancePerOutputName
from model_navigator.framework_api.commands.performance import ProfilingResults
from model_navigator.framework_api.common import DataObject, TensorMetadata
from model_navigator.framework_api.utils import JitType, RuntimeProvider, Status
from model_navigator.model import Format


@dataclass
class RuntimeResults(DataObject):
    runtime: RuntimeProvider
    status: Status
    err_msg: Dict[str, str]
    tolerance: Optional[TolerancePerOutputName] = None
    performance: Optional[List[ProfilingResults]] = None
    verified: bool = False

    @classmethod
    def from_dict(cls, dict: Mapping):
        return cls(
            runtime=RuntimeProvider(dict["runtime"]),
            status=Status(dict["status"]),
            tolerance=dict.get("tolerance"),
            performance=[ProfilingResults.from_dict(perf) for perf in dict.get("performance", [])]
            if dict["performance"] is not None
            else None,
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
            path=Path(dict["path"]) if dict.get("path") is not None in dict else None,
            runtime_results=[RuntimeResults.from_dict(runtime_results) for runtime_results in dict["runtime_results"]],
            torch_jit=JitType(dict["torch_jit"]) if dict.get("torch_jit") is not None else None,
            precision=TensorRTPrecision(dict["precision"]) if dict.get("precision") is not None else None,
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
            git_info=dict.get("git_info"),
            environment=dict["environment"],
            export_config=dict["export_config"],
            model_status=[ModelStatus.from_dict(model_status) for model_status in dict["model_status"]],
            input_metadata=TensorMetadata.from_json(dict["input_metadata"]) if dict.get("input_metadata") else None,
            output_metadata=TensorMetadata.from_json(dict["output_metadata"]) if dict.get("output_metadata") else None,
        )
