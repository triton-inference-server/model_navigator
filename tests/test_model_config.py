# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from typing import List, Optional

import tempfile
from pathlib import Path

import numpy as np
import pytest
from model_navigator import Format
from model_navigator.tensor import IOSpec, TensorSpec
from model_navigator.triton import ModelConfig
from tests.models_for_testings import ONNX_MODELS_URLS


@pytest.mark.parametrize(
    "model_url, inputs, outputs",
    [
        # (ONNX_MODELS_URLS["ResNet50"], TensorMetadata()),
        (
            ONNX_MODELS_URLS["MobileNetV2"],
            [TensorSpec("data", shape=(1, 3, 224, 224), dtype=np.dtype(np.float32))],
            [TensorSpec("mobilenetv20_output_flatten0_reshape0", shape=(1, 1000), dtype=np.dtype(np.float32))],
        )
    ],
)
def test_generating_of_io_for_onnx_model(
    datadir_mgr, model_url: str, inputs: Optional[List[TensorSpec]], outputs: Optional[List[TensorSpec]]
):
    max_batch_size = 16
    accelerator = "trt"
    precision = "fp16"

    *download_url, filename = model_url.split("/")
    download_url = "/".join(download_url)

    gunzip = filename.endswith("gz")
    datadir_mgr.download(download_url, files=[filename], scope="module", progressbar=True, gunzip=gunzip)
    model_path = datadir_mgr[filename]

    model_config = ModelConfig.create(
        model_path=model_path,
        model_name="foo",
        model_version="1",
        model_format="onnx",
        max_batch_size=max_batch_size,
        precision=precision,
        gpu_engine_count=2,
        preferred_batch_sizes=[max_batch_size // 2, max_batch_size],
        max_queue_delay_us=0,
        capture_cuda_graph=0,
        accelerator=accelerator,
    )

    assert model_config.inputs == inputs
    assert model_config.outputs == outputs


@pytest.mark.parametrize(
    "model_url, io_spec",
    [
        (
            ONNX_MODELS_URLS["MobileNetV2"],
            IOSpec(
                inputs={"data": TensorSpec("data", shape=(1, 3, 224, 224), dtype=np.dtype(np.float32))},
                outputs={
                    "mobilenetv20_output_flatten0_reshape0": TensorSpec(
                        "mobilenetv20_output_flatten0_reshape0", shape=(1, 1000), dtype=np.dtype(np.float32)
                    )
                },
            ),
        )
    ],
)
def test_generating_of_io_for_pyt(datadir_mgr, model_url: str, io_spec: IOSpec):
    max_batch_size = 16
    accelerator = "cuda"
    precision = "fp16"

    with tempfile.TemporaryDirectory() as dh:
        model_path = Path(dh) / "dummy.pt"
    spec_path = model_path.parent / f"{model_path.stem}.yaml"
    io_spec.write(spec_path)

    model_config = ModelConfig.create(
        model_path=model_path,
        model_name="foo",
        model_version="1",
        model_format=Format.TS_TRACE.value,
        max_batch_size=max_batch_size,
        precision=precision,
        gpu_engine_count=2,
        preferred_batch_sizes=[max_batch_size // 2, max_batch_size],
        max_queue_delay_us=0,
        capture_cuda_graph=0,
        accelerator=accelerator,
    )
    assert model_config.inputs == list(io_spec.inputs.values())
    assert model_config.outputs == list(io_spec.outputs.values())
