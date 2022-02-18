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

from ctypes import ArgumentError
from itertools import chain
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch  # pytype: disable=import-error
from transformers import AutoTokenizer, TensorType  # pytype: disable=import-error
from transformers.onnx import OnnxConfig  # pytype: disable=import-error

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.huggingface.onnx_config import get_onnx_config
from model_navigator.framework_api.huggingface.tasks import Task, get_automodel
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines import TorchPipelineManager
from model_navigator.framework_api.utils import (
    Framework,
    JitType,
    extract_input_shape,
    extract_output_shape,
    get_default_model_name,
    get_default_workdir,
)
from model_navigator.model import Format


class HFDataLoader:
    def __init__(self, model_name: str, onnx_config: OnnxConfig, device: str):

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._config = onnx_config
        self._device = device

    def __call__(self):

        yield {
            n: t.to(self._device)
            for n, t in self._config.generate_dummy_inputs(self._tokenizer, framework=TensorType.PYTORCH).items()
        }


def export(
    model_name: str,
    task: Optional[Task] = None,
    dataloader: Optional[Callable] = None,
    opset: Optional[int] = None,
    target_formats: Optional[Tuple[Format]] = None,
    jit_options: Optional[Tuple[JitType]] = None,
    workdir: Optional[Path] = None,
    override_workdir: bool = False,
    keep_workdir: bool = True,
    sample_count: Optional[int] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    onnx_config: Optional[OnnxConfig] = None,
    target_precisions: Optional[Tuple[TensorRTPrecision]] = None,
    save_data: bool = True,
) -> PackageDescriptor:
    """Function exports PyTorch model to all supported formats."""

    if model_name is None:
        model_name = get_default_model_name()
    if workdir is None:
        workdir = get_default_workdir()
    if target_formats is None:
        target_formats = (
            Format.TORCHSCRIPT,
            Format.ONNX,
            Format.TORCH_TRT,
            Format.TENSORRT,
        )
    if jit_options is None:
        jit_options = (
            JitType.SCRIPT,
            JitType.TRACE,
        )

    if sample_count is None:
        sample_count = 100

    if task is None:
        task = Task.BASE

    if target_precisions is None:
        target_precisions = (TensorRTPrecision.FP32, TensorRTPrecision.FP16)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_automodel(task).from_pretrained(model_name, torchscript=True)
    model.config.return_dict = True
    model.eval().to(device)

    if onnx_config is None:
        if task == Task.BASE:
            onnx_config = get_onnx_config(model)
        else:
            raise ArgumentError(f"OnnxConfig is required for model:  {model_name} on task: {task.value}")

    if opset is None:
        opset = onnx_config.default_onnx_opset

    if dataloader is None:
        dataloader = HFDataLoader(model_name, onnx_config, device)
        sample_count = 1

    if onnx_config.values_override is not None:
        LOGGER.info(f"Overriding {len(onnx_config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in onnx_config.values_override.items():
            LOGGER.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    input_names = tuple(onnx_config.inputs.keys())
    output_names = tuple(onnx_config.outputs.keys())
    dynamic_axes = {name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())}

    sample = next(dataloader())
    if isinstance(sample, dict):
        forward_kw_names = tuple(sample.keys())
    else:
        forward_kw_names = None

    input_metadata = extract_input_shape(dataloader, Framework.PYT)
    if input_names is not None:
        input_metadata = dict(zip(input_names, input_metadata.values()))

    output_metadata = extract_output_shape(model, dataloader, Framework.PYT)
    if output_names is not None:
        output_metadata = dict(zip(output_names, output_metadata.values()))

    config = Config(
        framework=Framework.PYT,
        model=model,
        model_name=model_name,
        dataloader=dataloader,
        target_formats=target_formats,
        target_jit_type=jit_options,
        opset=opset,
        workdir=workdir,
        override_workdir=override_workdir,
        keep_workdir=keep_workdir,
        sample_count=sample_count,
        atol=atol,
        rtol=rtol,
        dynamic_axes=dynamic_axes,
        target_precisions=target_precisions,
        save_data=save_data,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
        forward_kw_names=forward_kw_names,
    )

    pipeline_manager = TorchPipelineManager()
    return pipeline_manager.build(config)
