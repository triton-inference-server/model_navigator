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
from typing import Optional, Tuple, Union

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.framework_api.common import SizedDataLoader
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines import PipelineManager
from model_navigator.framework_api.pipelines.builders import (
    config_generation_builder,
    correctness_builder,
    onnx_export_builder,
    preprocessing_builder,
    profiling_builder,
)
from model_navigator.framework_api.utils import (
    Framework,
    RuntimeProvider,
    format2runtimes,
    get_default_max_workspace_size,
    get_default_model_name,
    get_default_workdir,
    parse_enum,
)
from model_navigator.model import Format


def export(
    model: Union[Path, str],
    dataloader: SizedDataLoader,
    target_precisions: Optional[Union[Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]]] = None,
    max_workspace_size: Optional[int] = None,
    minimum_segment_size: int = 3,
    model_name: Optional[str] = None,
    target_formats: Optional[Union[Union[str, Format], Tuple[Union[str, Format], ...]]] = None,
    workdir: Optional[Path] = None,
    override_workdir: bool = False,
    sample_count: Optional[int] = None,
    opset: Optional[int] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    disable_git_info: bool = False,
    batch_dim: Optional[int] = 0,
    onnx_runtimes: Optional[Union[Union[str, RuntimeProvider], Tuple[Union[str, RuntimeProvider], ...]]] = None,
    run_profiling: bool = True,
    profiler_config: Optional[ProfilerConfig] = None,
) -> PackageDescriptor:
    """Function exports ONNX model to all supported formats."""
    if isinstance(model, str):
        model = Path(model)
    if model_name is None:
        model_name = get_default_model_name()
    if workdir is None:
        workdir = get_default_workdir()
    if target_formats is None:
        target_formats = (
            Format.ONNX,
            Format.TENSORRT,
        )
    if max_workspace_size is None:
        max_workspace_size = get_default_max_workspace_size()
    if target_precisions is None:
        target_precisions = (TensorRTPrecision.FP32, TensorRTPrecision.FP16)
    if opset is None:
        opset = 14
    if sample_count is None:
        sample_count = 100

    if onnx_runtimes is None:
        onnx_runtimes = format2runtimes(Format.ONNX)

    if profiler_config is None:
        profiler_config = ProfilerConfig()

    target_formats, target_precisions, onnx_runtimes = (
        parse_enum(target_formats, Format),
        parse_enum(target_precisions, TensorRTPrecision),
        parse_enum(onnx_runtimes, RuntimeProvider),
    )
    config = Config(
        Framework.ONNX,
        model=model,
        model_name=model_name,
        dataloader=dataloader,
        target_precisions=target_precisions,
        max_workspace_size=max_workspace_size,
        minimum_segment_size=minimum_segment_size,
        workdir=workdir,
        override_workdir=override_workdir,
        target_formats=target_formats,
        sample_count=sample_count,
        opset=opset,
        atol=atol,
        rtol=rtol,
        disable_git_info=disable_git_info,
        batch_dim=batch_dim,
        onnx_runtimes=onnx_runtimes,
        profiler_config=profiler_config,
    )

    builders = [preprocessing_builder, onnx_export_builder, correctness_builder, config_generation_builder]
    if run_profiling:
        builders.append(profiling_builder)
    pipeline_manager = PipelineManager(builders)
    return PackageDescriptor.build(pipeline_manager, config)
