# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

from model_navigator import TensorRTProfile
from model_navigator.core.dataloader import expand_sample, load_samples
from model_navigator.core.tensor import TensorMetadata
from model_navigator.exceptions import ModelNavigatorRuntimeError


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
    batch_dim: Optional[int],
    target_device: str,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
    dataloader_max_batch_size: Optional[int] = None,
    device_max_batch_size: Optional[int] = None,
):
    """Export Torch model using dynamo.

    Args:
        exported_model_path: Output ONNX model path.
        input_metadata: List of input metadata.
        dataloader_trt_profile: Profiles generated based on shapes.
        batch_dim: Batch dimension.
        target_device: Device to load TorchScript model on.
        custom_args: Custom arguments passed to the export.
        navigator_workspace: Model Navigator workspace path. When None use current workdir.
        dataloader_max_batch_size: Maximum batch size in the dataloader. Defaults to None.
        device_max_batch_size: Maximum batch size that fits on the device. Defaults to None.
    """
    model = get_model()

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    conversion_sample = load_samples("conversion_samples", navigator_workspace, batch_dim)[0]
    input_metadata = TensorMetadata.from_json(input_metadata)

    max_batch_size = device_max_batch_size or dataloader_max_batch_size
    if max_batch_size is None:
        raise ModelNavigatorRuntimeError(
            "The provided `device_max_batch_size` and `dataloader_max_batch_size` are None."
        )

    # WAR for to big batch size value
    max_batch_size = max_batch_size if max_batch_size < 2048 else max_batch_size - 1

    # WAR to make data dynamic
    batch_size = min(2, max_batch_size)  # select the minimum value to expand samples
    expanded_sample = expand_sample(conversion_sample, input_metadata, batch_dim=batch_dim, batch_size=batch_size)

    dummy_input = {n: torch.from_numpy(val).to(target_device) for n, val in expanded_sample.items()}
    dummy_input = input_metadata.unflatten_sample(dummy_input, wrap_input=False)

    if not isinstance(dummy_input, tuple):
        dummy_input = (dummy_input,)
    if not isinstance(dummy_input[-1], dict):
        dummy_input = (*dummy_input, {})
    *args, kwargs = dummy_input

    # Collect trt profile for min and max shape data
    # FIXME: Use a common structure for the min/max shapes
    dataloader_trt_profile = TensorRTProfile.from_dict(dataloader_trt_profile)

    # Dynamic shapes support
    dynamic_shapes = []
    for name, spec_ in dataloader_trt_profile.items():
        tensor_metadata = input_metadata.get(name)
        if not tensor_metadata:
            continue

        dynamic_shapes_ = {}
        if max_batch_size > 1 and len(tensor_metadata.shape) > 0:
            dynamic_shapes_[0] = torch.export.Dim(f"{name}_batch", min=1, max=max_batch_size)

        for idx in range(1, len(spec_.min)):
            if spec_.min[idx] != spec_.max[idx]:
                dynamic_shapes_[idx] = torch.export.Dim(f"{name}__{idx}", min=spec_.min[idx], max=spec_.max[idx])

        dynamic_shapes.append(dynamic_shapes_)

    exported_model = torch.export.export(
        model,
        args=tuple(args),
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        **custom_args,
    )

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path
    torch.export.save(exported_model, exported_model_path.as_posix())
