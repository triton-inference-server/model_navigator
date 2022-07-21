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

from pathlib import Path
from typing import Optional, Union

from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines.builders import (
    config_generation_builder,
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
)
from model_navigator.framework_api.pipelines.pipeline_manager import PipelineManager
from model_navigator.framework_api.utils import Framework, get_framework_export_formats
from model_navigator.utils.device import get_gpus


def _copy_verified_staus(pkg_desc_from: PackageDescriptor, pkg_desc_to: PackageDescriptor):
    for model_status_1, model_status_2 in zip(
        pkg_desc_from.navigator_status.model_status, pkg_desc_to.navigator_status.model_status
    ):
        for runtime_status_1, runtime_status_2 in zip(model_status_1.runtime_results, model_status_2.runtime_results):
            runtime_status_2.verified = runtime_status_1.verified
    pkg_desc_to.save_status_file()


def _copy_git_info(pkg_desc_from: PackageDescriptor, pkg_desc_to: PackageDescriptor):
    pkg_desc_to.navigator_status.git_info = pkg_desc_from.navigator_status.git_info
    pkg_desc_to.save_status_file()


def load(
    path: Union[str, Path],
    workdir: Optional[Union[str, Path]] = None,
    override_workdir: bool = False,
    retest_conversions: bool = True,
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

    pkg_desc = PackageDescriptor.load(path=path, workdir=workdir, override_workdir=override_workdir)
    if not retest_conversions:
        return pkg_desc

    config = pkg_desc.config
    config.from_source = False
    config.override_workdir = False
    config.disable_git_info = True
    if profiler_config is not None:
        config.profiler_config = profiler_config
    if target_device is None:
        target_device = "cuda" if get_gpus(["all"]) else "cpu"
        LOGGER.info(f"Using `{target_device}` as target device.")
    config.target_device = target_device

    if pkg_desc.framework == Framework.PYT:
        from model_navigator.framework_api.pipelines.builders import torch_conversion_builder as conversion_builder
    elif pkg_desc.framework == Framework.TF2:
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
    _copy_verified_staus(pkg_desc, new_pkg_desc)
    _copy_git_info(pkg_desc, new_pkg_desc)
    return new_pkg_desc
