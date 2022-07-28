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
from typing import Dict, List, Optional, Sequence

from polygraphy.backend.trt import Profile

from model_navigator.__version__ import __version__
from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api._nav_package_format_version import NAV_PACKAGE_FORMAT_VERSION
from model_navigator.framework_api.commands.correctness import TolerancePerOutputName
from model_navigator.framework_api.commands.performance import ProfilerConfig, ProfilingResults
from model_navigator.framework_api.common import DataObject, TensorMetadata
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import (
    Framework,
    JitType,
    RuntimeProvider,
    Status,
    get_trt_profile_from_trt_dynamic_axes,
)
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
    def from_dict(cls, data_dict: Dict):
        return cls(
            runtime=RuntimeProvider(data_dict["runtime"]),
            status=Status(data_dict["status"]),
            tolerance=TolerancePerOutputName.from_json(data_dict.get("tolerance"))
            if data_dict.get("tolerance") is not None
            else None,
            performance=[ProfilingResults.from_dict(perf) for perf in data_dict["performance"]]
            if data_dict.get("performance") is not None
            else None,
            err_msg=data_dict.get("err_msg"),
            verified=data_dict["verified"],
        )


@dataclass
class ModelStatus(DataObject):
    format: Format
    path: Path
    runtime_results: List[RuntimeResults]
    torch_jit: Optional[JitType] = None
    precision: Optional[TensorRTPrecision] = None

    @classmethod
    def from_dict(cls, data_dict: Dict):
        return cls(
            format=Format(data_dict["format"]),
            path=Path(data_dict["path"]) if data_dict.get("path") is not None else None,
            runtime_results=[
                RuntimeResults.from_dict(runtime_results) for runtime_results in data_dict["runtime_results"]
            ],
            torch_jit=JitType(data_dict["torch_jit"]) if data_dict.get("torch_jit") is not None else None,
            precision=TensorRTPrecision(data_dict["precision"]) if data_dict.get("precision") is not None else None,
        )


@dataclass
class NavigatorStatus(DataObject):
    format_version: str
    model_navigator_version: str
    uuid: str
    git_info: Dict
    environment: Dict
    export_config: Dict
    model_status: List[ModelStatus]
    input_metadata: TensorMetadata
    output_metadata: TensorMetadata
    trt_profile: Profile

    @classmethod
    def from_dict(cls, data_dict: Dict):
        format_version = data_dict.get("format_version")
        if format_version == "0.1.0" and "profiler_config" in data_dict["export_config"]:
            format_version = "0.1.1"

        data_dict = DataDictUpdater.update(data_dict, format_version)

        if isinstance(data_dict["export_config"].get("_input_names"), Sequence):
            data_dict["export_config"]["_input_names"] = tuple(data_dict["export_config"]["_input_names"])
        if isinstance(data_dict["export_config"].get("_output_names"), Sequence):
            data_dict["export_config"]["_output_names"] = tuple(data_dict["export_config"]["_output_names"])

        trt_profile = Profile()
        for name, val in data_dict["trt_profile"].items():
            trt_profile.add(name, **val)

        return cls(
            format_version=NAV_PACKAGE_FORMAT_VERSION,
            model_navigator_version=__version__,
            uuid=data_dict["uuid"],
            git_info=data_dict["git_info"],
            environment=data_dict["environment"],
            export_config=data_dict["export_config"],
            model_status=[ModelStatus.from_dict(model_status) for model_status in data_dict["model_status"]],
            input_metadata=TensorMetadata.from_json(data_dict["input_metadata"]),
            output_metadata=TensorMetadata.from_json(data_dict["output_metadata"]),
            trt_profile=trt_profile,
        )


class DataDictUpdater:
    @staticmethod
    def update(data_dict: Dict, format_version: str):
        try:
            return getattr(DataDictUpdater, f"_update_navigator_status_v{format_version.replace('.', '_')}")(data_dict)
        except AttributeError:
            raise ValueError(f"Unknown .nav package format version: {format_version}")

    @staticmethod
    def _update_navigator_status_v0_1_0(data_dict: Dict):
        for model_status in data_dict["model_status"]:
            for runtime_results in model_status["runtime_results"]:
                for i in range(len(runtime_results.get("performance", []))):
                    perf_results = runtime_results["performance"][i]
                    runtime_results["performance"][i] = {
                        "batch_size": perf_results["batch_size"],
                        "avg_latency": perf_results["latency"],
                        "std_latency": None,
                        "p50_latency": None,
                        "p90_latency": None,
                        "p95_latency": None,
                        "p99_latency": None,
                        "throughput": perf_results["throughput"],
                        "request_count": None,
                    }

        if (
            Framework(data_dict["export_config"]["framework"]) == Framework.PYT
            and "precision_mode" not in data_dict["export_config"]
        ):
            default_val = TensorRTPrecisionMode.SINGLE.value
            LOGGER.info(f"Using default `precision_mode`: {default_val}")
            data_dict["export_config"]["precision_mode"] = default_val
        if "profiler_config" not in data_dict["export_config"]:
            default_val = ProfilerConfig().to_dict()
            LOGGER.info(f"Using default `profiler_config`: {default_val}")
            data_dict["export_config"]["profiler_config"] = default_val
        if "git_info" not in data_dict:
            data_dict["git_info"] = {}

        return DataDictUpdater._update_navigator_status_v0_1_1(data_dict)

    @staticmethod
    def _update_navigator_status_v0_1_1(data_dict: Dict):
        return DataDictUpdater._update_navigator_status_v0_1_2(data_dict)

    @staticmethod
    def _update_navigator_status_v0_1_2(data_dict: Dict):
        data_dict["trt_profile"] = DataObject.parse_value(
            get_trt_profile_from_trt_dynamic_axes(data_dict["export_config"]["trt_dynamic_axes"])
        )
        for model_status in data_dict["model_status"]:
            if model_status["format"] == "torch-trt" and model_status.get("precision") is None:
                model_status["precision"] = "fp32"
        return DataDictUpdater._update_navigator_status_v0_1_3(data_dict)

    @staticmethod
    def _update_navigator_status_v0_1_3(data_dict: Dict):
        if isinstance(data_dict["export_config"].get("_input_names"), Sequence):
            data_dict["export_config"]["_input_names"] = tuple(data_dict["export_config"]["_input_names"])
        if isinstance(data_dict["export_config"].get("_output_names"), Sequence):
            data_dict["export_config"]["_output_names"] = tuple(data_dict["export_config"]["_output_names"])
        return data_dict
