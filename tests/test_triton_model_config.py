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
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from model_navigator.model import Model, ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.triton import TritonModelConfigGenerator
from model_navigator.triton.config import (
    DeviceKind,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
    TritonModelSchedulerConfig,
)

CASE_TORCHSCRIPT_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES = (
    128,
    "model.pt",
    ModelSignatureConfig(
        inputs={"i__0": TensorSpec("i__0", shape=(-1, 3, 224, 224), dtype=np.dtype("float16"))},
        outputs={"o__1": TensorSpec("o__1", shape=(-1, 1000), dtype=np.dtype("float16"))},
    ),
)

CASE_TORCHSCRIPT_SIMPLE_IMAGE_MODEL_WITH_DYNAMIC_AXES = (
    128,
    "model.pt",
    ModelSignatureConfig(
        inputs={"i__0": TensorSpec("i__0", shape=(-1, 3, -1, -1), dtype=np.dtype("float16"))},
        outputs={"o__1": TensorSpec("o__1", shape=(-1, 1000), dtype=np.dtype("float16"))},
    ),
)


@pytest.mark.parametrize(
    "max_batch_size,model_filename,signature",
    [CASE_TORCHSCRIPT_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES, CASE_TORCHSCRIPT_SIMPLE_IMAGE_MODEL_WITH_DYNAMIC_AXES],
)
def test_model_config_parsing_signature_for_torchscript(monkeypatch, max_batch_size, model_filename, signature):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # create dummy triton model repo structure
        model_path = temp_dir / "1" / model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w"):
            pass

        config_path = temp_dir / "config.pbtxt"

        src_model = Model("dummy", model_path, signature_if_missing=signature)
        optimization_config = TritonModelOptimizationConfig()
        scheduler_config = TritonModelSchedulerConfig(max_batch_size=max_batch_size)
        instances_config = TritonModelInstancesConfig({DeviceKind.GPU: 1})
        initial_model_config_generator = TritonModelConfigGenerator(
            src_model,
            optimization_config=optimization_config,
            scheduler_config=scheduler_config,
            instances_config=instances_config,
        )
        initial_model_config_generator.save_config_pbtxt(config_path)

        parsed_model_config_generator = TritonModelConfigGenerator.from_triton_config_pbtxt(config_path)
        assert parsed_model_config_generator.model.signature == src_model.signature
