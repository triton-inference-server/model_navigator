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

from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

# pytype: disable=import-error
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, TensorType
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.onnx import OnnxConfig

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.framework_api.common import SizedDataLoader
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.contrib.huggingface.datasets import (
    HFDataLoaderFactory,
    get_default_preprocess_function,
)
from model_navigator.framework_api.contrib.huggingface.onnx_config import get_onnx_config
from model_navigator.framework_api.contrib.huggingface.torch.task import TASK_OUTPUTS_MAPPING, get_task_from_model
from model_navigator.framework_api.contrib.huggingface.torch.utils import (
    get_max_sequence_length,
    get_pretrained_model_from_config,
)
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines.builders import (
    config_generation_builder,
    correctness_builder,
    preprocessing_builder,
    profiling_builder,
    torch_export_builder,
)
from model_navigator.framework_api.pipelines.pipeline_manager import PipelineManager
from model_navigator.framework_api.utils import (
    Framework,
    JitType,
    RuntimeProvider,
    format2runtimes,
    get_default_max_workspace_size,
    get_default_workdir,
    parse_enum,
)
from model_navigator.model import Format

# pytype: enable=import-error


class HFDataLoader:
    def __init__(self, tokenizer, onnx_config: OnnxConfig, device: str, max_sequence_length: Optional[int] = None):

        self._tokenizer = tokenizer
        self._config = onnx_config
        self._device = device
        self._max_sequence_length = max_sequence_length

    def __call__(self):
        return [self._config.generate_dummy_inputs(self._tokenizer, framework=TensorType.PYTORCH) for _ in range(10)]


def _get_dataloader(
    *,
    dataset_name,
    tokenizer,
    onnx_config,
    target_device,
    padding,
    max_sequence_len,
    max_bs,
    dataset_preprocessing_function,
):
    if isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
        tokenizer.pad_token = tokenizer.eos_token
    if dataset_name is None:
        return HFDataLoader(tokenizer, onnx_config, target_device, max_sequence_length=max_sequence_len)()
    else:
        dataset = load_dataset(dataset_name)["train"]
        if dataset_preprocessing_function is None:
            dataset_preprocessing_function = get_default_preprocess_function(dataset_name, tokenizer, max_sequence_len)
        dataloader_factory = HFDataLoaderFactory(
            dataset,
            tokenizer,
            dataset_preprocessing_function,
            list(onnx_config.inputs.keys()),
            target_device,
            padding=padding,
            max_sequence_length=max_sequence_len,
        )
        return dataloader_factory(max_bs)


def export(
    model_name: str,
    dataloader: Optional[SizedDataLoader] = None,
    dataset_name: Optional[str] = None,
    dataset_preprocessing_function: Optional[Callable] = None,
    opset: Optional[int] = None,
    target_formats: Optional[Union[Union[str, Format], Tuple[Union[str, Format], ...]]] = None,
    jit_options: Optional[Union[Union[str, JitType], Tuple[Union[str, JitType], ...]]] = None,
    workdir: Optional[Path] = None,
    override_workdir: bool = False,
    sample_count: Optional[int] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    onnx_config: Optional[OnnxConfig] = None,
    target_precisions: Optional[Union[Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]]] = None,
    trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
    max_workspace_size: Optional[int] = None,
    target_device: Optional[str] = None,
    disable_git_info: bool = False,
    max_bs: int = 1,
    batch_dim: Optional[int] = 0,
    padding: Union[bool, str] = True,
    max_sequence_len: Optional[int] = None,
    onnx_runtimes: Optional[Union[Union[str, RuntimeProvider], Tuple[Union[str, RuntimeProvider], ...]]] = None,
    run_profiling: bool = True,
    profiler_config: Optional[ProfilerConfig] = None,
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

    config = AutoConfig.from_pretrained(model_name)
    model = get_pretrained_model_from_config(model_name, config, torchscript=True)
    model.config.return_dict = True

    task = get_task_from_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if max_sequence_len is None:
        max_sequence_len = get_max_sequence_length(tokenizer)
    if onnx_config is None:
        onnx_config = get_onnx_config(model.config)
    if opset is None:
        opset = onnx_config.default_onnx_opset

    if dataloader is None:
        dataloader = _get_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            onnx_config=onnx_config,
            target_device=target_device,
            padding=padding,
            max_sequence_len=max_sequence_len,
            max_bs=max_bs,
            dataset_preprocessing_function=dataset_preprocessing_function,
        )

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
    sample = next(iter(dataloader))
    if isinstance(sample, Mapping):
        forward_kw_names = tuple(sample.keys())
    else:
        forward_kw_names = None
    onnx_runtimes = onnx_runtimes or format2runtimes(Format.ONNX)

    if profiler_config is None:
        profiler_config = ProfilerConfig()

    target_formats, jit_options, target_precisions, onnx_runtimes = (
        parse_enum(target_formats, Format),
        parse_enum(jit_options, JitType),
        parse_enum(target_precisions, TensorRTPrecision),
        parse_enum(onnx_runtimes, RuntimeProvider),
    )
    config = Config(
        framework=Framework.PYT,
        model=model,
        model_name=model_name.replace("/", "-"),
        dataloader=dataloader,
        target_formats=target_formats,
        target_jit_type=jit_options,
        opset=opset,
        workdir=workdir,
        override_workdir=override_workdir,
        sample_count=sample_count,
        atol=atol,
        rtol=rtol,
        dynamic_axes=dynamic_axes,
        target_precisions=target_precisions,
        _input_names=input_names,
        _output_names=output_names,
        forward_kw_names=forward_kw_names,
        max_workspace_size=max_workspace_size,
        trt_dynamic_axes=trt_dynamic_axes,
        target_device=target_device,
        disable_git_info=disable_git_info,
        batch_dim=batch_dim,
        onnx_runtimes=onnx_runtimes,
        profiler_config=profiler_config,
    )

    builders = [preprocessing_builder, torch_export_builder, correctness_builder, config_generation_builder]
    if run_profiling:
        builders.append(profiling_builder)
    pipeline_manager = PipelineManager(builders)
    return PackageDescriptor.build(pipeline_manager, config)
