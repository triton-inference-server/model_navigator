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

import pathlib
from typing import Any, Dict, Optional

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
    batch_dim: Optional[int],
    target_device: str,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
):
    """Export Torch model using dynamo.

    Args:
        exported_model_path (str): Output ONNX model path.
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

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]
    input_metadata = TensorMetadata.from_json(input_metadata)

    dummy_input = {n: torch.from_numpy(val).to(target_device) for n, val in profiling_sample.items()}
    dummy_input = input_metadata.unflatten_sample(dummy_input, wrap_input=False)

    exported_model = torch.onnx.dynamo_export(
        model,
        dummy_input,
        **custom_args,
    )

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path
    exported_model.save(exported_model_path.as_posix())
