# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from typing import Callable, Optional, Tuple

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines import TFPipelineManager
from model_navigator.framework_api.utils import (
    Framework,
    get_default_max_workspace_size,
    get_default_model_name,
    get_default_workdir,
)
from model_navigator.model import Format


def export(
    model,
    dataloader: Callable,
    target_precisions: Optional[Tuple[TensorRTPrecision]] = None,
    max_workspace_size: Optional[int] = None,
    minimum_segment_size: int = 3,
    model_name: Optional[str] = None,
    target_formats: Optional[Tuple[Format]] = None,
    workdir: Optional[Path] = None,
    override_workdir: bool = False,
    keep_workdir: bool = True,
    sample_count: Optional[int] = None,
    opset: Optional[int] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    save_data: bool = True,
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
        )
    if max_workspace_size is None:
        max_workspace_size = get_default_max_workspace_size()
    if target_precisions is None:
        target_precisions = (TensorRTPrecision.FP32, TensorRTPrecision.FP16)
    if opset is None:
        opset = 14
    if sample_count is None:
        sample_count = 100

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
        keep_workdir=keep_workdir,
        target_formats=target_formats,
        sample_count=sample_count,
        opset=opset,
        atol=atol,
        rtol=rtol,
        save_data=save_data,
    )

    pipeline_manager = TFPipelineManager()
    return pipeline_manager.build(config)
