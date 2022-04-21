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
from model_navigator.framework_api.common import SizedDataLoader
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines import TFPipelineManager
from model_navigator.framework_api.utils import (
    Framework,
    get_default_max_workspace_size,
    get_default_model_name,
    get_default_workdir,
    parse_enum,
)
from model_navigator.model import Format


def export(
    model,
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
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    disable_git_info: bool = False,
    batch_dim: Optional[int] = 0,
) -> PackageDescriptor:
    """Function exports TensorFlow 2 model to all supported formats."""
    if model_name is None:
        model_name = get_default_model_name()
    if workdir is None:
        workdir = get_default_workdir()
    if target_formats is None:
        target_formats = (
            Format.TF_SAVEDMODEL,
            Format.TF_TRT,
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

    target_formats, target_precisions = parse_enum(target_formats, Format), parse_enum(
        target_precisions, TensorRTPrecision
    )
    config = Config(
        Framework.TF2,
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
        _input_names=input_names,
        _output_names=output_names,
        disable_git_info=disable_git_info,
        batch_dim=batch_dim,
    )

    pipeline_manager = TFPipelineManager()
    return pipeline_manager.build(config)
