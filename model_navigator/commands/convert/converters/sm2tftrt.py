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
"""Script that converts SavedModel to Tensorflow-TensorRT model."""

import pathlib
from typing import Any, Dict, Optional

import fire
from tensorflow.python.compiler.tensorrt import trt_convert as trtc  # pytype: disable=import-error

from model_navigator.core.dataloader import expand_sample, load_samples
from model_navigator.core.tensor import TensorMetadata


def convert(
    exported_model_path: str,
    converted_model_path: str,
    max_workspace_size: int,
    target_precision: str,
    minimum_segment_size: int,
    batch_dim: int,
    max_batch_size: int,
    custom_args: Dict[str, Any],
    input_metadata: Dict,
    navigator_workspace: Optional[str] = None,
) -> None:
    """Convert SavedModel to Tensorflow-TensorRT model.

    For detailed explanation of TensorRT arguments please refer to [documentation]
    [documentation]: https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html


    Args:
        exported_model_path: Path of SavedModel.
        converted_model_path: Output path of Tensorflow-TensorRT model.
        max_workspace_size: TensorRT maximum workspace size in bytes.
        target_precision: TensorRT precision. Could be "fp32" or "fp16".
        minimum_segment_size: TensorRT minimum segment size.
        batch_dim: Batch dimension.
        max_batch_size: Maximum batch size.
        input_metadata: Dict of input metadata.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            If None use current workdir. Defaults to None.
        custom_args (Optional[Dict[str, Any]], optional): Passthrough parameters for TrtGraphConverterV2
            For available arguments check TensorRT documentation: https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html

    Yields:
        _type_: _description_
    """
    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    conversion_samples = load_samples("conversion_samples", navigator_workspace, batch_dim)

    input_metadata = TensorMetadata.from_json(input_metadata)

    def _dataloader():
        for sample in conversion_samples:
            yield sample
            yield expand_sample(sample, input_metadata, batch_dim, max_batch_size)

    params = trtc.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=max_workspace_size,
        precision_mode=target_precision,
        minimum_segment_size=minimum_segment_size,
    )

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path

    # TODO: allow setting dynamic_shape_profile_strategy

    converter = trtc.TrtGraphConverterV2(
        input_saved_model_dir=exported_model_path.as_posix(),
        use_dynamic_shape=True,
        conversion_params=params,
        **custom_args,
    )

    converter.convert()
    converter.build(_dataloader)

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = navigator_workspace / converted_model_path

    converter.save(converted_model_path.as_posix())


if __name__ == "__main__":
    fire.Fire(convert)
