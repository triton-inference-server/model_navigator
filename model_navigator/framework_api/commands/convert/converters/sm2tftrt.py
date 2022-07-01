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


import fire
from tensorflow.python.compiler.tensorrt import trt_convert as trtc  # pytype: disable=import-error

from model_navigator.framework_api.utils import load_samples, sample_to_tuple


def convert(
    exported_model_path: str,
    converted_model_path: str,
    max_workspace_size: int,
    target_precision: str,
    minimum_segment_size: int,
    package_path: str,
    batch_dim: int,
):

    conversion_samples = load_samples("conversion_samples", package_path, batch_dim)

    def _dataloader():
        for sample in conversion_samples:
            yield sample_to_tuple(sample)

    params = trtc.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=max_workspace_size,
        precision_mode=target_precision,
        minimum_segment_size=minimum_segment_size,
    )
    # TODO: allow setting dynamic_shape_profile_strategy
    converter = trtc.TrtGraphConverterV2(
        input_saved_model_dir=exported_model_path, use_dynamic_shape=True, conversion_params=params
    )

    converter.convert()
    converter.build(_dataloader)
    converter.save(converted_model_path)


if __name__ == "__main__":
    fire.Fire(convert)
