# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import pathlib
from typing import Optional

import fire
from tensorflow.python.compiler.tensorrt import trt_convert as trtc  # pytype: disable=import-error

from model_navigator.framework_api.utils import load_samples


def convert(
    exported_model_path: str,
    converted_model_path: str,
    max_workspace_size: int,
    target_precision: str,
    minimum_segment_size: int,
    batch_dim: int,
    navigator_workdir: Optional[str] = None,
):

    if not navigator_workdir:
        navigator_workdir = pathlib.Path.cwd()
    navigator_workdir = pathlib.Path(navigator_workdir)

    conversion_samples = load_samples("conversion_samples", navigator_workdir, batch_dim)

    def _dataloader():
        yield from conversion_samples

    params = trtc.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=max_workspace_size,
        precision_mode=target_precision,
        minimum_segment_size=minimum_segment_size,
    )

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workdir / exported_model_path

    # TODO: allow setting dynamic_shape_profile_strategy

    converter = trtc.TrtGraphConverterV2(
        input_saved_model_dir=exported_model_path.as_posix(), use_dynamic_shape=True, conversion_params=params
    )

    converter.convert()
    converter.build(_dataloader)

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = navigator_workdir / converted_model_path

    converter.save(converted_model_path.as_posix())


if __name__ == "__main__":
    fire.Fire(convert)
