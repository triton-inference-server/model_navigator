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

import gc
import inspect
import logging
import pathlib
from typing import Any, Dict, List, Mapping, Optional

import fire
import onnx
import onnx_graphsurgeon as gs
import torch  # pytype: disable=import-error

from model_navigator.configuration import TensorRTProfile
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
    input_metadata: Dict[str, Any],
    input_names: List[str],
    output_names: List[str],
    batch_dim: Optional[int],
    target_device: str,
    verbose: bool = False,
    custom_args: Optional[Dict[str, Any]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    dynamic_shapes: Optional[bool] = None,
    opset: Optional[int] = None,
    navigator_workspace: Optional[str] = None,
    dataloader_max_batch_size: Optional[int] = None,
    dataloader_trt_profile: Optional[Dict[str, Any]] = None,
    device_max_batch_size: Optional[int] = None,
    export_engine: Optional[str] = None,
):
    """Export Torch model to ONNX model.

    Args:
        exported_model_path (str): Path to save the exported model.
        input_metadata (Dict[str, Any]): Input metadata.
        input_names (List[str]): Input names.
        output_names (List[str]): Output names.
        batch_dim (Optional[int]): Batch dimension.
        target_device (str): Device to export the model to.
        verbose (bool): Verbose mode.
        custom_args (Dict[str, Any]): Custom arguments.
        dynamic_axes (Dict[str, Dict[int, str]]): Dynamic axes.
        dynamic_shapes (Optional[bool]): Dynamic shapes.
        opset (Optional[int]): ONNX opset.
        navigator_workspace (Optional[str]): Navigator workspace.
        dataloader_max_batch_size (Optional[int]): Maximum batch size in the dataloader.
        dataloader_trt_profile (Optional[Dict[str, Any]]): Profiles generated based on shapes.
        device_max_batch_size (Optional[int]): Maximum batch size that fits on the device.
        export_engine (str): Export engine to use.
    """
    export_engine = export_engine or "torch-trace"
    custom_args = custom_args or {}

    if export_engine == "torch-trace":
        export_using_trace(
            navigator_workspace=navigator_workspace,
            exported_model_path=exported_model_path,
            opset=opset,
            input_metadata=input_metadata,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            batch_dim=batch_dim,
            target_device=target_device,
            custom_args=custom_args,
        )
    elif export_engine == "torch-dynamo":
        export_using_dynamo(
            navigator_workspace=navigator_workspace,
            exported_model_path=exported_model_path,
            opset=opset,
            input_metadata=input_metadata,
            input_names=input_names,
            output_names=output_names,
            batch_dim=batch_dim,
            target_device=target_device,
            dynamic_shapes=dynamic_shapes,
            verbose=verbose,
            custom_args=custom_args,
            dataloader_trt_profile=dataloader_trt_profile,
            dataloader_max_batch_size=dataloader_max_batch_size,
            device_max_batch_size=device_max_batch_size,
        )


def export_using_trace(
    exported_model_path: str,
    opset: int,
    input_metadata: Dict[str, Any],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    batch_dim: Optional[int],
    target_device: str,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
) -> None:
    """Export Torch model to ONNX model using torch.onnx.export.

    Args:
        exported_model_path (str): Output ONNX model path.
        opset (int): ONNX opset.
        input_metadata (Dict): List of input metadata.
        input_names (List[str]): List of model input names.
        output_names (List[str]): List of model output names.
        dynamic_axes (Dict[str, Dict[int, str]]): Configuration of the dynamic axes.
        batch_dim (Optional[int]): Batch dimension.
        target_device (str): Device to export model to ONNX on.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.export
            For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
    """
    model = get_model()
    model.to(target_device)

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

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

    dummy_input = input_metadata.unflatten_sample(dummy_input)

    # torch.onnx.export requires inputs to be a tuple or tensor
    if isinstance(dummy_input, Mapping):
        dummy_input = (dummy_input,)

    args_mapping, kwargs_mapping = input_metadata.pytree_metadata.get_names_mapping()

    # Use inspect.signature instead of getfullargspec for more complete parameter information
    forward_signature = inspect.signature(model.forward)
    forward_params = list(forward_signature.parameters.keys())

    args_count = len(args_mapping)
    forward_kwargs = forward_params[args_count:]

    for argname in kwargs_mapping:
        assert argname in forward_kwargs, f"Argument {argname} is not in forward argspec."

    input_names = []
    for args_names in args_mapping:
        input_names.extend(args_names)

    for argname in forward_kwargs:
        if argname in kwargs_mapping:
            input_names.extend(kwargs_mapping[argname])

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path

    try:
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
    finally:
        for tensor in dummy_input_map.values():
            tensor.cpu()

        del dummy_input_map
        gc.collect()
        torch.cuda.empty_cache()


def export_using_dynamo(
    exported_model_path: str,
    opset: int,
    input_metadata: Dict[str, Any],
    dataloader_trt_profile: Dict[str, Any],
    input_names: List[str],
    output_names: List[str],
    batch_dim: Optional[int],
    target_device: str,
    dynamic_shapes: bool,
    verbose: bool,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
    dataloader_max_batch_size: Optional[int] = None,
    device_max_batch_size: Optional[int] = None,
):
    """Export Torch model to ONNX model using torch.onnx.export with dynamo=True .

    Args:
        exported_model_path (str): Output ONNX model path.
        opset (int): ONNX opset.
        input_metadata (Dict[str, Any]): List of input metadata.
        dataloader_trt_profile: Profiles generated based on shapes.
        input_names (List[str]): List of model input names.
        output_names (List[str]): List of model output names.
        batch_dim (Optional[int]): Batch dimension.
        target_device (str): Device to load TorchScript model on.
        dynamic_shapes (bool): Enable dynamic shapes.
        verbose (bool): Enable verbose mode.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.jit.trace
            For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/jit.html#torch.jit.trace
        dataloader_max_batch_size: Maximum batch size in the dataloader. Defaults to None.
        device_max_batch_size: Maximum batch size that fits on the device. Defaults to None.
    """
    model = get_model()
    model.to(target_device)

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    input_metadata = TensorMetadata.from_json(input_metadata)

    def expand_batch_dim(tensor, batch_dim, max_batch_size):
        if batch_dim is not None and tensor.shape[batch_dim] < max_batch_size:
            expand_shape = list(tensor.shape)
            expand_shape[batch_dim] = max_batch_size
            expanded_tensor = tensor.expand(*expand_shape)
            return expanded_tensor
        return tensor

    dummy_input = {n: torch.from_numpy(val).to(target_device) for n, val in profiling_sample.items()}
    dummy_input = input_metadata.unflatten_sample(dummy_input, wrap_input=False)

    if not isinstance(dummy_input, tuple):
        dummy_input = (dummy_input,)
    if not isinstance(dummy_input[-1], dict):
        dummy_input = (*dummy_input, {})
    *args, kwargs = dummy_input

    # Expand batch_dim of tensors to max_batch_size
    max_batch_size = device_max_batch_size or dataloader_max_batch_size
    if max_batch_size is not None:
        args = tuple(
            expand_batch_dim(arg, batch_dim, max_batch_size) if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        kwargs = {
            k: expand_batch_dim(v, batch_dim, max_batch_size) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

    loglevel = logging.WARNING if verbose else logging.ERROR

    root_logger = logging.getLogger()
    original_loglevel = root_logger.getEffectiveLevel()
    root_logger.setLevel(loglevel)

    # Dynamic shapes support

    # Collect trt profile for min and max shape data
    # FIXME: Use a common structure for the min/max shapes
    dataloader_trt_profile = TensorRTProfile.from_dict(dataloader_trt_profile)
    dynamic_shapes = []
    for name, spec_ in dataloader_trt_profile.items():
        tensor_metadata = input_metadata.get(name)
        if not tensor_metadata:
            continue

        dynamic_shape_map = {}
        if max_batch_size is not None and max_batch_size > 1 and len(tensor_metadata.shape) > 0:
            dynamic_shape_map[0] = torch.export.Dim("batch", min=1, max=max_batch_size)

        for idx in range(1, len(spec_.min)):
            if spec_.min[idx] != spec_.max[idx]:
                dynamic_shape_map[idx] = torch.export.Dim(f"{name}__{idx}", min=spec_.min[idx], max=spec_.max[idx])

        dynamic_shapes.append(dynamic_shape_map)

    try:
        exported_model = torch.onnx.export(
            model,
            args=tuple(args),
            kwargs=kwargs,
            **custom_args,
            # opset_version=opset, # do not set for dynamo
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            fallback=False,
        )

        exported_model_path = pathlib.Path(exported_model_path)
        if not exported_model_path.is_absolute():
            exported_model_path = navigator_workspace / exported_model_path
        exported_model.save(exported_model_path.as_posix())
    finally:
        root_logger.setLevel(original_loglevel)
        # Offload tensors to CPU
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg.cpu()
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                value.cpu()

        del args
        del kwargs
        gc.collect()
        torch.cuda.empty_cache()

    _modify_onnx_io_names(exported_model_path, input_names, output_names, exported_model_path)


def _modify_onnx_io_names(model_path, new_input_names, new_output_names, output_path):
    # Load the ONNX model
    graph = gs.import_onnx(onnx.load(model_path))

    # Check if the number of new input names matches the number of inputs in the graph
    if len(new_input_names) != len(graph.inputs):
        raise ValueError("Number of new input names must match the number of inputs in the ONNX graph.")

    # Modify the input names
    for i, input in enumerate(graph.inputs):
        input.name = new_input_names[i]

    # Check if the number of new output names matches the number of outputs in the graph
    if len(new_output_names) != len(graph.outputs):
        raise ValueError("Number of new output names must match the number of outputs in the ONNX graph.")

    # Modify the output names
    for i, output in enumerate(graph.outputs):
        output.name = new_output_names[i]

    # Save the modified model
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), output_path)


if __name__ == "__main__":
    fire.Fire(export)
