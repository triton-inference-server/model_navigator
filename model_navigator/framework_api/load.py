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

from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines.builders import (
    config_generation_builder,
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
)
from model_navigator.framework_api.pipelines.pipeline_manager import PipelineManager
from model_navigator.framework_api.utils import Framework, JitType, parse_enum
from model_navigator.model import Format


def _copy_verified_staus(pkg_desc_1: PackageDescriptor, pkg_desc_2: PackageDescriptor):
    for model_status_1, model_status_2 in zip(
        pkg_desc_1.navigator_status.model_status, pkg_desc_2.navigator_status.model_status
    ):
        for runtime_status_1, runtime_status_2 in zip(model_status_1.runtime_results, model_status_2.runtime_results):
            runtime_status_2.verified = runtime_status_1.verified
    pkg_desc_2.save_status_file()


def load(
    path: Union[str, Path],
    workdir: Optional[Union[str, Path]] = None,
    override_workdir: bool = False,
    retest_conversions: bool = True,
    run_profiling: Optional[bool] = None,
    profiler_config: Optional[ProfilerConfig] = None,
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

    saved_config = pkg_desc.navigator_status.export_config

    if profiler_config is None:
        profiler_config = ProfilerConfig()

    config = Config(
        framework=pkg_desc.framework,
        model_name=pkg_desc.model_name,
        model=None,
        dataloader=[],
        workdir=pkg_desc.workdir,
        override_workdir=False,
        target_formats=parse_enum(saved_config["target_formats"], Format),
        sample_count=saved_config["sample_count"],
        batch_dim=saved_config["batch_dim"],
        seed=saved_config["seed"],
        from_source=False,
        target_precisions=parse_enum(saved_config["target_precisions"], TensorRTPrecision),
        precision_mode=parse_enum(
            saved_config.get("precision_mode", TensorRTPrecisionMode.SINGLE), TensorRTPrecisionMode
        )[0],
        target_jit_type=parse_enum(saved_config.get("target_jit_type"), JitType),
        target_device=saved_config["target_device"],
        opset=saved_config["opset"],
        max_batch_size=saved_config["max_batch_size"],
        trt_dynamic_axes=saved_config["trt_dynamic_axes"],
        dynamic_axes=saved_config["dynamic_axes"],
        max_workspace_size=saved_config.get("max_workspace_size"),
        minimum_segment_size=saved_config.get("minimum_segment_size"),
        disable_git_info=True,
        profiler_config=profiler_config,
    )
    if pkg_desc.framework == Framework.PYT:
        from model_navigator.framework_api.pipelines.builders import torch_export_builder as export_builder
    elif pkg_desc.framework == Framework.TF2:
        from model_navigator.framework_api.pipelines.builders import tensorflow_export_builder as export_builder
    else:  # ONNX
        from model_navigator.framework_api.pipelines.builders import onnx_export_builder as export_builder

    builders = [preprocessing_builder, export_builder, correctness_builder, config_generation_builder]
    if run_profiling:
        builders.append(profiling_builder)
    pipeline_manager = PipelineManager(builders)
    new_pkg_desc = PackageDescriptor.build(pipeline_manager, config)
    _copy_verified_staus(pkg_desc, new_pkg_desc)
    return new_pkg_desc
