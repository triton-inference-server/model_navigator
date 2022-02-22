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

from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch  # pytype: disable=import-error
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, TensorType
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.onnx import OnnxConfig

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.huggingface.datasets import HFDataLoaderFactory, get_default_preprocess_function
from model_navigator.framework_api.huggingface.onnx_config import get_onnx_config
from model_navigator.framework_api.huggingface.task import TASK_OUTPUTS_MAPPING, get_task_from_model
from model_navigator.framework_api.huggingface.utils import get_max_sequence_length, get_pretrained_model_from_config
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines import TorchPipelineManager
from model_navigator.framework_api.utils import Framework, JitType, get_default_max_workspace_size, get_default_workdir
from model_navigator.model import Format


class HFDataLoader:
    def __init__(self, tokenizer, onnx_config: OnnxConfig, device: str, max_sequence_length: Optional[int] = None):

        self._tokenizer = tokenizer
        self._config = onnx_config
        self._device = device
        self._max_sequence_length = max_sequence_length

    def __call__(self):
        for _ in range(100):
            yield {
                n: t
                for n, t in self._config.generate_dummy_inputs(self._tokenizer, framework=TensorType.PYTORCH).items()
            }


def export(
    model_name: str,
    dataloader: Optional[Callable] = None,
    dataset_name: Optional[str] = None,
    dataset_preprocessing_function: Optional[Callable] = None,
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
    trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
    save_data: bool = True,
    max_workspace_size: Optional[int] = None,
    target_device: Optional[str] = None,
    disable_git_info: bool = False,
) -> PackageDescriptor:
    """Function exports PyTorch model to all supported formats."""
    if workdir is None:
        workdir = get_default_workdir()
    if max_workspace_size is None:
        max_workspace_size = get_default_max_workspace_size()

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

    if target_precisions is None:
        target_precisions = (TensorRTPrecision.FP32, TensorRTPrecision.FP16)

    if target_device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    config = config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torchscript=True)
    model = get_pretrained_model_from_config(config, torchscript=True)
    model.config.return_dict = True

    task = get_task_from_model(model)
    max_seq_len = get_max_sequence_length(config)

    if onnx_config is None:
        onnx_config = get_onnx_config(model.config)

    if opset is None:
        opset = onnx_config.default_onnx_opset

    if dataloader is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            tokenizer.pad_token = tokenizer.eos_token
        if dataset_name is None:
            dataloader = HFDataLoader(tokenizer, onnx_config, target_device, max_sequence_length=max_seq_len)
        else:
            dataset = load_dataset(dataset_name)["train"]
            if dataset_preprocessing_function is None:
                dataset_preprocessing_function = get_default_preprocess_function(dataset_name, tokenizer, max_seq_len)
            dataloader_factory = HFDataLoaderFactory(
                dataset, tokenizer, dataset_preprocessing_function, onnx_config, target_device
            )
            dataloader = dataloader_factory(1)

    if onnx_config.values_override is not None:
        LOGGER.info(f"Overriding {len(onnx_config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in onnx_config.values_override.items():
            LOGGER.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    inputs = onnx_config.inputs
    outputs = TASK_OUTPUTS_MAPPING.get(task, onnx_config.outputs)
    input_names = tuple(inputs.keys())
    output_names = tuple(outputs.keys())
    dynamic_axes = {name: axes for name, axes in chain(inputs.items(), outputs.items())}
    model.eval().to(target_device)
    sample = next(iter(dataloader()))
    if isinstance(sample, dict):
        forward_kw_names = tuple(sample.keys())
    else:
        forward_kw_names = None

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
        _input_names=input_names,
        _output_names=output_names,
        forward_kw_names=forward_kw_names,
        max_workspace_size=max_workspace_size,
        trt_dynamic_axes=trt_dynamic_axes,
        target_device=target_device,
        disable_git_info=disable_git_info,
    )

    pipeline_manager = TorchPipelineManager()
    return pipeline_manager.build(config)
