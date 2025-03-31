# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Export Torch model using dynamo."""

import logging
import pathlib
from typing import Any, Dict, List, Optional

import onnx
import onnx_graphsurgeon as gs
import torch  # pytype: disable=import-error

from model_navigator.configuration import TensorRTProfile
from model_navigator.core.dataloader import load_samples
from model_navigator.core.tensor import TensorMetadata


def get_model() -> torch.nn.Module:
    """Get model instance.

    Returns:
        Model to be exported.
    """
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def export(
    exported_model_path: str,
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
    """Export Torch model using dynamo.

    Args:
        exported_model_path (str): Output ONNX model path.
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

        dynamic_shapes_ = {}
        if max_batch_size is not None and max_batch_size > 1 and len(tensor_metadata.shape) > 0:
            dynamic_shapes_[0] = torch.export.Dim("batch", min=1, max=max_batch_size)

        for idx in range(1, len(spec_.min)):
            if spec_.min[idx] != spec_.max[idx]:
                dynamic_shapes_[idx] = torch.export.Dim(f"{name}__{idx}", min=spec_.min[idx], max=spec_.max[idx])

        dynamic_shapes.append(dynamic_shapes_)

    try:
        exported_model = torch.onnx.export(
            model,
            args=tuple(args),
            kwargs=kwargs,
            **custom_args,
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
