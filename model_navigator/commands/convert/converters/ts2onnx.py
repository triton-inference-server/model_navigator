# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Convert TorchScript model to ONNX model."""

import pathlib
from typing import Any, Dict, List, Optional

import fire
import torch  # pytype: disable=import-error

from model_navigator.core.dataloader import load_samples
from model_navigator.core.tensor import TensorMetadata


def convert(
    exported_model_path: str,
    converted_model_path: str,
    opset: int,
    input_metadata: Dict,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    batch_dim: Optional[int],
    target_device: str,
    custom_args: Dict[str, Any],
    workspace: Optional[str] = None,
):
    """Run TorchScript to ONNX conversion.

    Args:
        exported_model_path (str): TorchScript model path.
        converted_model_path (str): Output ONNX model path.
        opset (int): ONNX opset.
        input_metadata (Dict): Input metadata.
        input_names (List[str]): List of model input names.
        output_names (List[str]): List of model output names.
        dynamic_axes (Dict[str, Dict[int, str]]): Configuration of the dynamic axes.
        batch_dim (Optional[int]): Batch dimension.
        target_device (str): Device to load TorchScript model on.
        workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for torch.onnx.export
            For available arguments check PyTorch documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
    """
    if not workspace:
        workspace = pathlib.Path.cwd()
    workspace = pathlib.Path(workspace)

    profiling_sample = load_samples("profiling_sample", workspace, batch_dim)[0]

    inputs = {name: torch.from_numpy(val).to(target_device) for name, val in profiling_sample.items()}
    input_metadata = TensorMetadata.from_json(input_metadata)
    dummy_input = input_metadata.unflatten_sample(inputs, wrap_input=True)

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = workspace / exported_model_path

    model = torch.jit.load(exported_model_path.as_posix(), map_location=target_device)
    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = workspace / converted_model_path

    torch.onnx.export(
        model,
        args=dummy_input,
        f=converted_model_path.as_posix(),
        verbose=False,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        **custom_args,
    )


if __name__ == "__main__":
    fire.Fire(convert)
