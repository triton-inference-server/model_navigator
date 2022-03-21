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
from typing import Callable, Optional, Tuple, Union

import tensorflow as tf  # pytype: disable=import-error
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, TensorType
from transformers.models.gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from transformers.onnx.config import OnnxConfig

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.common import SizedDataLoader
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.contrib.huggingface.datasets import (
    HFDataLoaderFactory,
    get_default_preprocess_function,
)
from model_navigator.framework_api.contrib.huggingface.onnx_config import get_onnx_config
from model_navigator.framework_api.contrib.huggingface.tensorflow.task import TASK_OUTPUTS_MAPPING, get_task_from_model
from model_navigator.framework_api.contrib.huggingface.tensorflow.utils import (
    get_max_sequence_length,
    get_pretrained_model_from_config,
)
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines import TFPipelineManager
from model_navigator.framework_api.utils import (
    Framework,
    get_default_max_workspace_size,
    get_default_workdir,
    parse_enum,
)
from model_navigator.model import Format


class HFDataLoader:
    def __init__(self, tokenizer, onnx_config: OnnxConfig, device: str, max_sequence_length: Optional[int] = None):

        self._tokenizer = tokenizer
        self._config = onnx_config
        self._device = device
        self._max_sequence_length = max_sequence_length

    def __call__(self):
        return [self._config.generate_dummy_inputs(self._tokenizer, framework=TensorType.TENSORFLOW) for _ in range(10)]


def export(
    model_name: str,
    dataloader: Optional[SizedDataLoader] = None,
    dataset_name: Optional[str] = None,
    dataset_preprocessing_function: Optional[Callable] = None,
    opset: Optional[int] = None,
    target_formats: Optional[Union[Union[str, Format], Tuple[Union[str, Format], ...]]] = None,
    workdir: Optional[Path] = None,
    override_workdir: bool = False,
    keep_workdir: bool = True,
    sample_count: Optional[int] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    onnx_config: Optional[OnnxConfig] = None,
    target_precisions: Optional[Union[Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]]] = None,
    save_data: bool = True,
    max_workspace_size: Optional[int] = None,
    minimum_segment_size: int = 3,
    target_device: Optional[str] = None,
    disable_git_info: bool = False,
    max_bs: int = 1,
    batch_dim: Optional[int] = 0,
    padding: Union[bool, str] = True,
    max_sequence_len: Optional[int] = None,
) -> PackageDescriptor:

    config = AutoConfig.from_pretrained(model_name)
    model = get_pretrained_model_from_config(config)
    model.config.return_dict = True

    if max_sequence_len is None:
        max_sequence_len = get_max_sequence_length(config)

    if onnx_config is None:
        onnx_config = get_onnx_config(model.config)

    inputs = onnx_config.inputs

    task = get_task_from_model(model)
    outputs = TASK_OUTPUTS_MAPPING.get(task, onnx_config.outputs)
    input_names = tuple(inputs.keys())
    output_names = tuple(outputs.keys())

    if dataloader is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            tokenizer.pad_token = tokenizer.eos_token
        if dataset_name is None:
            # raise NotImplementedError
            dataloader = HFDataLoader(tokenizer, onnx_config, target_device, max_sequence_length=max_sequence_len)()
        else:
            dataset = load_dataset(dataset_name)["train"]
            if dataset_preprocessing_function is None:
                dataset_preprocessing_function = get_default_preprocess_function(
                    dataset_name, tokenizer, max_sequence_len
                )
            dataloader_factory = HFDataLoaderFactory(
                dataset,
                tokenizer,
                dataset_preprocessing_function,
                list(onnx_config.inputs.keys()),
                # list(model._saved_model_inputs_spec.keys()),
                # list(inspect.signature(model.call).parameters.keys()),
                target_device,
                padding=padding,
                max_sequence_length=max_sequence_len,
                return_tensors=TensorType.TENSORFLOW,
            )
            dataloader = dataloader_factory(max_bs, framework=Framework.TF2)

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

    sample = next(iter(dataloader))
    input_spec = {
        input_name: tf.keras.Input(
            shape=tuple(None for d in list(tensor.shape)[1:]), dtype=tensor.dtype, name=input_name
        )
        for input_name, tensor in sample.items()
    }
    model._saved_model_inputs_spec = None
    model._set_save_spec(input_spec)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

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
        keep_workdir=keep_workdir,
        target_formats=target_formats,
        sample_count=sample_count,
        opset=opset,
        atol=atol,
        rtol=rtol,
        save_data=save_data,
        disable_git_info=disable_git_info,
        batch_dim=batch_dim,
        _input_names=input_names,
        _output_names=output_names,
    )

    pipeline_manager = TFPipelineManager()
    return pipeline_manager.build(config)
