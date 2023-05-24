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
"""Export Torch model to TorchScript model."""

import pathlib
from typing import Optional

import fire
import torch  # pytype: disable=import-error

from model_navigator.api.config import JitType
from model_navigator.utils.dataloader import load_samples


def get_model() -> torch.nn.Module:
    """Get model instance.

    Returns:
        Model to be exported.
    """
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def export(
    exported_model_path: str,
    target_jit_type: str,
    batch_dim: Optional[int],
    target_device: str,
    strict: bool,
    navigator_workspace: Optional[str] = None,
):
    """Export Torch model to ONNX model.

    Args:
        exported_model_path (str): Output ONNX model path.
        target_jit_type (str): TorchScript jit type. Could be "trace" or "script".
        batch_dim (Optional[int]): Batch dimension.
        target_device (str): Device to load TorchScript model on.
        strict (bool): Enable or Disable strict flag for tracer used in TorchScript export.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
    """
    model = get_model()
    target_jit_type = JitType(target_jit_type)

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    profiling_sample = load_samples("profiling_sample", navigator_workspace, batch_dim)[0]

    if target_jit_type == JitType.SCRIPT:
        script_module = torch.jit.script(model)
    else:
        dummy_input = tuple(torch.from_numpy(val).to(target_device) for val in profiling_sample.values())
        script_module = torch.jit.trace(model, dummy_input, strict=strict)

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path

    torch.jit.save(script_module, exported_model_path.as_posix())


if __name__ == "__main__":
    fire.Fire(export)
