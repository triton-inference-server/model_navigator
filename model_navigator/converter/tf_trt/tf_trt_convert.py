#!/usr/bin/env python3
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
import logging
from pathlib import Path

# pytype: disable=import-error
from tensorflow.python.compiler.tensorrt import trt_convert as trtc

# pytype: enable=import-error
from model_navigator.converter.dataloader import Dataloader

LOGGER = logging.getLogger("tf_trt_converter")


def wrap_dataloader(dataloader):
    """TF-TRT Converter changes _arg_keywords (as of 22.02) (WHY????),
    so we cannot rely on our dataloder-supplied names.
    Return the sample in positional format instead.
    """

    def _wrap():
        for sample in dataloader:
            yield tuple(sample.values())

    return _wrap


def convert_tf2(
    input_path: Path,
    output_path: Path,
    max_workspace_size: int,
    precision: str,
    dataloader: Dataloader,
    minimum_segment_size: int = 3,
):
    """Optimize a Tensorflow 2.x Savedmodel at `input_path`
    with TF-TRT.
    Store the resulting SavedModel at `output_path`.
    """
    params = trtc.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=max_workspace_size,
        precision_mode=precision,
        minimum_segment_size=minimum_segment_size,
    )
    # TODO: check if model signature has dynamic shapes
    # TODO: allow setting dynamic_shape_profile_strategy
    converter = trtc.TrtGraphConverterV2(
        input_saved_model_dir=input_path.as_posix(),
        conversion_params=params,
        use_dynamic_shape=True,
    )

    converter.convert()

    if dataloader:
        LOGGER.info("Pre-building TRT engines.")
        converter.build(wrap_dataloader(dataloader))

    converter.save(output_path.as_posix())
