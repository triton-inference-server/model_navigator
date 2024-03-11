# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Export Torch model to ONNX model."""

import inspect
import pathlib
from typing import Any, Dict, List, Mapping, Optional

import fire
import torch  # pytype: disable=import-error

from model_navigator.core.dataloader import load_samples
from model_navigator.core.tensor import TensorMetadata
from model_navigator.utils.common import numpy_to_torch_dtype


def get_model() -> torch.nn.Module:
    """Get model instance.

    Returns:
        Model to be exported.
    """
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def export(
    exported_model_path: str,
    opset: int,
    input_metadata: Dict[str, Any],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    batch_dim: Optional[int],
    export_device: str,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
) -> None:
    """Export Torch model to ONNX model.

    Args:
        exported_model_path (str): Output ONNX model path.
        opset (int): ONNX opset.
        input_metadata (Dict): List of input metadata.
        input_names (List[str]): List of model input names.
        output_names (List[str]): List of model output names.
        dynamic_axes (Dict[str, Dict[int, str]]): Configuration of the dynamic axes.
        batch_dim (Optional[int]): Batch dimension.
        export_device (str): Device to export model to ONNX on.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.export
            For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
    """
    model = get_model()
    model = model.to(export_device)

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    input_metadata = TensorMetadata.from_json(input_metadata)

    dummy_input = {n: torch.from_numpy(val).to(export_device) for n, val in profiling_sample.items()}

    # adjust input dtypes to match input_metadata
    # TODO: Remove when torch.bfloat16 will be supported
    for n, spec in input_metadata.items():
        if not isinstance(spec.dtype, torch.dtype):
            torch_dtype = numpy_to_torch_dtype(spec.dtype)
        else:
            torch_dtype = spec.dtype
        dummy_input[n] = dummy_input[n].to(torch_dtype)

    dummy_input = input_metadata.unflatten_sample(dummy_input)

    # torch.onnx.export requires inputs to be a tuple or tensor
    if isinstance(dummy_input, Mapping):
        dummy_input = (dummy_input,)

    forward_argspec = inspect.getfullargspec(model.forward)
    forward_args = forward_argspec.args[1:]

    args_mapping, kwargs_mapping = input_metadata.pytree_metadata.get_names_mapping()

    for argname in kwargs_mapping:
        assert argname in forward_args, f"Argument {argname} is not in forward argspec."

    input_names = []
    for args_names in args_mapping:
        input_names.extend(args_names)

    for argname in forward_args:
        if argname in kwargs_mapping:
            input_names.extend(kwargs_mapping[argname])

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path

    torch.onnx.export(
        model,
        args=dummy_input,
        f=exported_model_path.as_posix(),
        verbose=False,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        **custom_args,
    )


if __name__ == "__main__":
    fire.Fire(export)
