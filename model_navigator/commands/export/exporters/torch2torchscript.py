# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
"""Export Torch model to TorchScript model."""

import gc
import inspect
import pathlib
from typing import Any, Dict, Optional

import fire
import torch  # pytype: disable=import-error

from model_navigator.configuration import JitType
from model_navigator.core.dataloader import load_samples
from model_navigator.core.tensor import TensorMetadata
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.frameworks.torch.utils import offload_torch_model_to_cpu
from model_navigator.utils.common import numpy_to_torch_dtype


def get_model() -> torch.nn.Module:
    """Get model instance.

    Returns:
        Model to be exported.
    """
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def export(
    exported_model_path: str,
    target_jit_type: str,
    input_metadata: Dict[str, Any],
    batch_dim: Optional[int],
    target_device: str,
    strict: bool,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
):
    """Export Torch model to ONNX model.

    Args:
        exported_model_path (str): Output ONNX model path.
        target_jit_type (str): TorchScript jit type. Could be "trace" or "script".
        input_metadata (Dict[str, Any]): List of input metadata.
        batch_dim (Optional[int]): Batch dimension.
        target_device (str): Device to load TorchScript model on.
        strict (bool): Enable or Disable strict flag for tracer used in TorchScript export.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.jit.trace
            For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/jit.html#torch.jit.trace
    """
    model = get_model()
    model.to(target_device)

    target_jit_type = JitType(target_jit_type)

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    if target_jit_type == JitType.SCRIPT:
        _export_script(model, exported_model_path, navigator_workspace)
    else:
        _export_trace(
            model,
            exported_model_path,
            input_metadata,
            batch_dim,
            target_device,
            navigator_workspace,
            strict,
            custom_args,
        )


def _export_script(model, exported_model_path, navigator_workspace):
    script_module = None
    try:
        script_module = torch.jit.script(model)
        exported_model_path = pathlib.Path(exported_model_path)
        if not exported_model_path.is_absolute():
            exported_model_path = navigator_workspace / exported_model_path

        torch.jit.save(script_module, exported_model_path.as_posix())
    finally:
        if script_module is not None:
            offload_torch_model_to_cpu(script_module)


def _export_trace(
    model, exported_model_path, input_metadata, batch_dim, target_device, navigator_workspace, strict, custom_args
):
    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    input_metadata = TensorMetadata.from_json(input_metadata)

    dummy_input_map = {n: torch.from_numpy(val).to(target_device) for n, val in profiling_sample.items()}

    # adjust input dtypes to match input_metadata
    # TODO: Remove when torch.bfloat16 will be supported
    dummy_input = {}
    for n, spec in input_metadata.items():
        if not isinstance(spec.dtype, torch.dtype):
            torch_dtype = numpy_to_torch_dtype(spec.dtype)
        else:
            torch_dtype = spec.dtype
        dummy_input[n] = dummy_input_map[n].to(torch_dtype)

    dummy_input = input_metadata.unflatten_sample(dummy_input, wrap_input=True)
    if isinstance(dummy_input[-1], dict):
        args, kwargs = dummy_input[:-1], dummy_input[-1]
    else:
        args, kwargs = dummy_input, {}

    if args:
        if kwargs:
            try:
                forward_signature = inspect.signature(model.forward)
                forward_args = list(forward_signature.parameters.keys())
                for i, arg in enumerate(args):
                    kwargs[forward_args[i]] = arg
                input_kwargs = {
                    "example_kwarg_inputs": kwargs,
                }
            except BaseException as e:
                raise ModelNavigatorUserInputError(
                    "TorchScript trace does not support both args and kwargs and Model Navigator was unable to convert the input to kwargs. Please provide the input as only kwargs or only args."
                ) from e
        else:
            input_kwargs = {
                "example_inputs": args,
            }
    else:
        input_kwargs = {
            "example_kwarg_inputs": kwargs,
        }

    script_module = None
    try:
        script_module = torch.jit.trace(model, strict=strict, **input_kwargs, **custom_args)
        exported_model_path = pathlib.Path(exported_model_path)
        if not exported_model_path.is_absolute():
            exported_model_path = navigator_workspace / exported_model_path

        torch.jit.save(script_module, exported_model_path.as_posix())
    finally:
        if script_module is not None:
            offload_torch_model_to_cpu(script_module)
            del script_module

        for tensor in dummy_input_map.values():
            tensor.cpu()

        del dummy_input_map
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(export)
