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

import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from packaging import version

from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.framework_api.common import DataObject, TensorMetadata
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.exceptions import ModelNavigatorBackwardCompatibilityError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines.builders import (
    config_generation_builder,
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
)
from model_navigator.framework_api.pipelines.pipeline_manager import PipelineManager
from model_navigator.framework_api.status import NavigatorStatus
from model_navigator.framework_api.utils import (
    Framework,
    format2runtimes,
    get_default_workdir,
    get_framework_export_formats,
    get_package_path,
    get_trt_profile_from_trt_dynamic_axes,
)
from model_navigator.model import Format
from model_navigator.utils.device import get_gpus


class StatusDictUpdater:
    def __init__(self):
        self._updates = {
            version.parse("0.1.0"): self._update_from_v0_1_0,
            version.parse("0.1.2"): self._update_from_v0_1_2,
            version.parse("0.1.3"): self._update_from_v0_1_3,
        }

    @staticmethod
    def _update_from_v0_1_0(data_dict: Dict):
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

    @staticmethod
    def _update_from_v0_1_2(data_dict: Dict):
        data_dict["trt_profile"] = DataObject.parse_value(
            get_trt_profile_from_trt_dynamic_axes(data_dict["export_config"]["trt_dynamic_axes"])
        )
        for model_status in data_dict["model_status"]:
            if model_status["format"] == "torch-trt" and model_status.get("precision") is None:
                model_status["precision"] = "fp32"

    @staticmethod
    def _update_from_v0_1_3(data_dict: Dict):
        if (
            Framework(data_dict["export_config"]["framework"]) == Framework.PYT
            and data_dict["export_config"].get("precision_mode") is None
        ):
            default_val = TensorRTPrecisionMode.SINGLE.value
            LOGGER.info(f"Using default `precision_mode`: {default_val}")
            data_dict["export_config"]["precision_mode"] = default_val
        if "onnx_runtimes" in data_dict["export_config"]:
            data_dict["export_config"]["runtimes"] = data_dict["export_config"].pop("onnx_runtimes")

    def update_(self, data_dict: Dict, format_version: version.Version):
        for update_from_version, update_func in self._updates.items():
            if format_version <= update_from_version:
                update_func(data_dict)


class PackageUpdater:
    def __init__(self):
        self._updates = {version.parse("0.3.3"): self._update_from_v0_3_3}

    @staticmethod
    def _update_from_v0_3_3(pkg_desc):
        if pkg_desc.framework == Framework.TF2:
            if len(pkg_desc.navigator_status.input_metadata) > 1:
                raise ModelNavigatorBackwardCompatibilityError(
                    "Cannot load TensorFlow2 .nav packages generated by Model Navigator version < 0.3.4 and with multiple inputs."
                )
            _update_savedmodel_signature(
                model_name=pkg_desc.config.model_name,
                input_metadata=pkg_desc.navigator_status.input_metadata,
                output_metadata=pkg_desc.navigator_status.output_metadata,
                workdir=pkg_desc.workdir,
            )

    def update_(self, package_descriptor: PackageDescriptor, pkg_version: version.Version):
        for update_from_version, update_func in self._updates.items():
            if pkg_version <= update_from_version:
                update_func(package_descriptor)


def _copy_verified_status(pkg_desc_from: PackageDescriptor, pkg_desc_to: PackageDescriptor):
    for model_status_1, model_status_2 in zip(
        pkg_desc_from.navigator_status.model_status, pkg_desc_to.navigator_status.model_status
    ):
        for runtime_status_1, runtime_status_2 in zip(model_status_1.runtime_results, model_status_2.runtime_results):
            runtime_status_2.verified = runtime_status_1.verified
    pkg_desc_to.save_status_file()


def _copy_git_info(pkg_desc_from: PackageDescriptor, pkg_desc_to: PackageDescriptor):
    pkg_desc_to.navigator_status.git_info = pkg_desc_from.navigator_status.git_info
    pkg_desc_to.save_status_file()


def _update_savedmodel_signature(
    model_name: str, input_metadata: TensorMetadata, output_metadata: TensorMetadata, workdir: Path
):
    LOGGER.info("Updating SavedModel signature...")
    from model_navigator.framework_api.commands.export.tf import UpdateSavedModelSignature

    cmd = UpdateSavedModelSignature()
    cmd(model_name, input_metadata, output_metadata, workdir)


def _load_package_descriptor(
    path: Union[str, Path],
    workdir: Optional[Union[str, Path]] = None,
    override_workdir: bool = False,
) -> PackageDescriptor:
    def _filter_out_converted_models(paths: List[str], pkg_desc: PackageDescriptor):
        export_formats = get_framework_export_formats(pkg_desc.framework)
        converted_model_paths = []
        for model_status in pkg_desc.navigator_status.model_status:
            if model_status.format not in export_formats:
                if model_status.path:
                    converted_model_paths.append(model_status.path.as_posix())
        return [p for p in paths if not p.startswith(tuple(converted_model_paths))]

    def _extract_pkg_version(status_dict):
        return version.parse(status_dict.get("model_navigator_version", "0.3.0"))

    def _extract_format_version(status_dict):
        format_version = status_dict.get("format_version", "0.1.0")
        if format_version == "0.1.0" and "profiler_config" in status_dict["export_config"]:
            format_version = "0.1.1"
        return version.parse(format_version)

    path = Path(path)
    if workdir is None:
        workdir = get_default_workdir()
    workdir = Path(workdir)

    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(PackageDescriptor.status_filename) as status_file:
            status_dict = yaml.safe_load(status_file)
        pkg_version = _extract_pkg_version(status_dict)
        format_version = _extract_format_version(status_dict)
        StatusDictUpdater().update_(status_dict, format_version)
        navigator_status = NavigatorStatus.from_dict(status_dict)

        package_path = get_package_path(workdir=workdir, model_name=navigator_status.export_config["model_name"])
        if package_path.exists():
            if override_workdir:
                shutil.rmtree(package_path)
            else:
                raise FileExistsError(package_path)

        pkg_desc = PackageDescriptor(navigator_status, workdir)
        all_members = zf.namelist()
        filtered_members = _filter_out_converted_models(all_members, pkg_desc)
        zf.extractall(package_path, members=filtered_members)

    PackageUpdater().update_(pkg_desc, pkg_version)
    pkg_desc.save_status_file()
    return pkg_desc


def _update_config_defaults(config: Config, framework: Framework):
    config.target_precisions = (TensorRTPrecision.FP32, TensorRTPrecision.FP16)
    config.precision_mode = TensorRTPrecisionMode.HIERARCHY
    config.runtimes = format2runtimes(Format.ONNX)

    if framework == Framework.PYT:
        config.target_formats = (Format.TORCHSCRIPT, Format.ONNX, Format.TORCH_TRT, Format.TENSORRT)
    elif framework == Framework.TF2:
        config.target_formats = (Format.TF_SAVEDMODEL, Format.TF_TRT, Format.ONNX, Format.TENSORRT)
    elif framework == Framework.JAX:
        config.target_formats = (Format.TF_SAVEDMODEL, Format.TF_TRT, Format.ONNX, Format.TENSORRT)
    else:
        assert framework == Framework.ONNX
        config.target_formats = (Format.ONNX, Format.TENSORRT)


def load(
    path: Union[str, Path],
    workdir: Optional[Union[str, Path]] = None,
    override_workdir: bool = False,
    retest_conversions: bool = True,
    use_config_defaults: bool = True,
    run_profiling: Optional[bool] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    target_device: Optional[str] = None,
) -> PackageDescriptor:
    """Load .nav package from the path.
    If `retest_conversions = True` rerun conversion tests (including correctness and performance).
    """
    if run_profiling is None:
        run_profiling = retest_conversions
    if not retest_conversions and run_profiling:
        raise ValueError("Cannot run profiling without retesting conversions.")

    pkg_desc = _load_package_descriptor(path=path, workdir=workdir, override_workdir=override_workdir)
    if not retest_conversions:
        return pkg_desc

    config = pkg_desc.config
    config.from_source = False
    config.override_workdir = False
    config.disable_git_info = True

    if use_config_defaults:
        _update_config_defaults(config, pkg_desc.framework)

    if profiler_config is not None:
        config.profiler_config = profiler_config
    if target_device is None:
        target_device = "cuda" if get_gpus(["all"]) else "cpu"
        LOGGER.info(f"Using `{target_device}` as target device.")
    config.target_device = target_device

    if pkg_desc.framework == Framework.PYT:
        from model_navigator.framework_api.pipelines.builders import torch_conversion_builder as conversion_builder
    elif pkg_desc.framework in (Framework.TF2, Framework.JAX):
        from model_navigator.framework_api.pipelines.builders import tensorflow_conversion_builder as conversion_builder
    else:
        assert pkg_desc.framework == Framework.ONNX
        from model_navigator.framework_api.pipelines.builders import onnx_conversion_builder as conversion_builder

    builders = [preprocessing_builder, conversion_builder, correctness_builder, config_generation_builder]
    if run_profiling:
        builders.append(profiling_builder)
    exported_model_status = [
        model_status
        for model_status in pkg_desc.navigator_status.model_status
        if model_status.format in get_framework_export_formats(pkg_desc.framework)
    ]
    pipeline_manager = PipelineManager(builders)
    new_pkg_desc = PackageDescriptor.build(pipeline_manager, config, existing_model_status=exported_model_status)
    _copy_verified_status(pkg_desc, new_pkg_desc)
    _copy_git_info(pkg_desc, new_pkg_desc)
    return new_pkg_desc
