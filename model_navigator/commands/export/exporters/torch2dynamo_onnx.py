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
    input_names: List[str],
    output_names: List[str],
    batch_dim: Optional[int],
    target_device: str,
    dynamic_shapes: bool,
    verbose: bool,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
):
    """Export Torch model using dynamo.

    Args:
        exported_model_path (str): Output ONNX model path.
        input_metadata (Dict[str, Any]): List of input metadata.
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
    """
    model = get_model()

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    input_metadata = TensorMetadata.from_json(input_metadata)

    dummy_input = {n: torch.from_numpy(val).to(target_device) for n, val in profiling_sample.items()}
    dummy_input = input_metadata.unflatten_sample(dummy_input, wrap_input=False)

    if not isinstance(dummy_input, tuple):
        dummy_input = (dummy_input,)
    if not isinstance(dummy_input[-1], dict):
        dummy_input = (*dummy_input, {})
    *args, kwargs = dummy_input

    loglevel = logging.INFO if verbose else logging.WARNING
    export_options_kwargs = {}
    export_options_kwargs["op_level_debug"] = verbose
    export_options_kwargs["diagnostic_options"] = torch.onnx.DiagnosticOptions(verbosity_level=loglevel)
    if dynamic_shapes:
        export_options_kwargs["dynamic_shapes"] = True
    export_options = torch.onnx.ExportOptions(**export_options_kwargs)

    root_logger = logging.getLogger()
    root_loglevel = root_logger.getEffectiveLevel()
    root_logger.setLevel(loglevel)

    exported_model = torch.onnx.dynamo_export(
        model,
        *args,
        **custom_args,
        **kwargs,
        export_options=export_options,
    )

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path
    exported_model.save(exported_model_path.as_posix())

    root_logger.setLevel(root_loglevel)

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
