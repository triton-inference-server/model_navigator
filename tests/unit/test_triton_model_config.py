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

from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.model import Model, ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.triton.config import (
    Batching,
    DeviceKind,
    TritonBatchingConfig,
    TritonCustomBackendParametersConfig,
    TritonDynamicBatchingConfig,
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
)
from model_navigator.triton.model_config import TritonModelConfigGenerator

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

CASE_TENSORRT_PLAN_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES = (
    128,
    "model.plan",
    ModelSignatureConfig(
        inputs={"i__0": TensorSpec("i__0", shape=(-1, 3, 224, 224), dtype=np.dtype("float16"))},
        outputs={"o__1": TensorSpec("o__1", shape=(-1, 1000), dtype=np.dtype("float16"))},
    ),
)

CASE_TENSORRT_PLAN_IMAGE_MODEL_WITH_DYNAMIC_AXES = (
    128,
    "model.plan",
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
        batching_config = TritonBatchingConfig(max_batch_size=max_batch_size)
        optimization_config = TritonModelOptimizationConfig()
        tensorrt_common_config = TensorRTCommonConfig()
        dynamic_batching_config = TritonDynamicBatchingConfig()
        instances_config = TritonModelInstancesConfig({DeviceKind.GPU: 1})
        backend_parameters_config = TritonCustomBackendParametersConfig()
        initial_model_config_generator = TritonModelConfigGenerator(
            src_model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            tensorrt_common_config=tensorrt_common_config,
            dynamic_batching_config=dynamic_batching_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )
        initial_model_config_generator.save(config_path)

        parsed_model_config_generator = TritonModelConfigGenerator.parse_triton_config_pbtxt(config_path)
        assert parsed_model_config_generator.model.signature == src_model.signature
        assert parsed_model_config_generator.optimization_config == optimization_config
        assert parsed_model_config_generator.dynamic_batching_config == dynamic_batching_config
        assert parsed_model_config_generator.instances_config == instances_config


@pytest.mark.parametrize(
    "max_batch_size,model_filename,signature",
    [CASE_TENSORRT_PLAN_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES, CASE_TENSORRT_PLAN_IMAGE_MODEL_WITH_DYNAMIC_AXES],
)
def test_model_config_parsing_signature_for_tensorrt_plan(monkeypatch, max_batch_size, model_filename, signature):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # create dummy triton model repo structure
        model_path = temp_dir / "1" / model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w"):
            pass

        config_path = temp_dir / "config.pbtxt"

        src_model = Model("dummy", model_path, signature_if_missing=signature)
        batching_config = TritonBatchingConfig(max_batch_size=max_batch_size)
        optimization_config = TritonModelOptimizationConfig()
        tensorrt_common_config = TensorRTCommonConfig()
        dynamic_batching_config = TritonDynamicBatchingConfig()
        instances_config = TritonModelInstancesConfig({DeviceKind.GPU: 1})
        backend_parameters_config = TritonCustomBackendParametersConfig()
        initial_model_config_generator = TritonModelConfigGenerator(
            src_model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            tensorrt_common_config=tensorrt_common_config,
            dynamic_batching_config=dynamic_batching_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )
        initial_model_config_generator.save(config_path)

        parsed_model_config_generator = TritonModelConfigGenerator.parse_triton_config_pbtxt(config_path)
        assert parsed_model_config_generator.model.signature == src_model.signature
        assert parsed_model_config_generator.optimization_config == optimization_config
        assert parsed_model_config_generator.dynamic_batching_config == dynamic_batching_config
        assert parsed_model_config_generator.instances_config == instances_config
        # assert parsed_model_config_generator.backend_parameters_config == backend_parameters_config


@pytest.mark.parametrize(
    "max_batch_size,model_filename,signature",
    [CASE_TENSORRT_PLAN_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES, CASE_TENSORRT_PLAN_IMAGE_MODEL_WITH_DYNAMIC_AXES],
)
def test_model_config_parsing_signature_with_static_batching(monkeypatch, max_batch_size, model_filename, signature):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # create dummy triton model repo structure
        model_path = temp_dir / "1" / model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w"):
            pass

        config_path = temp_dir / "config.pbtxt"

        src_model = Model("dummy", model_path, signature_if_missing=signature)
        batching_config = TritonBatchingConfig(max_batch_size=max_batch_size, batching=Batching.STATIC)
        optimization_config = TritonModelOptimizationConfig()
        tensorrt_common_config = TensorRTCommonConfig()
        dynamic_batching_config = TritonDynamicBatchingConfig()
        instances_config = TritonModelInstancesConfig({DeviceKind.GPU: 1})
        backend_parameters_config = TritonCustomBackendParametersConfig()
        initial_model_config_generator = TritonModelConfigGenerator(
            src_model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            tensorrt_common_config=tensorrt_common_config,
            dynamic_batching_config=dynamic_batching_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )
        initial_model_config_generator.save(config_path)

        parsed_model_config_generator = TritonModelConfigGenerator.parse_triton_config_pbtxt(config_path)
        assert parsed_model_config_generator.batching_config == batching_config
        assert parsed_model_config_generator.model.signature == src_model.signature
        assert parsed_model_config_generator.optimization_config == optimization_config
        assert parsed_model_config_generator.dynamic_batching_config == dynamic_batching_config
        assert parsed_model_config_generator.instances_config == instances_config


@pytest.mark.parametrize(
    "max_batch_size,model_filename,signature",
    [CASE_TENSORRT_PLAN_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES, CASE_TENSORRT_PLAN_IMAGE_MODEL_WITH_DYNAMIC_AXES],
)
def test_model_config_parsing_signature_with_disabled_batching(monkeypatch, max_batch_size, model_filename, signature):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # create dummy triton model repo structure
        model_path = temp_dir / "1" / model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w"):
            pass

        config_path = temp_dir / "config.pbtxt"

        src_model = Model("dummy", model_path, signature_if_missing=signature)
        batching_config = TritonBatchingConfig(max_batch_size=max_batch_size, batching=Batching.DISABLED)
        optimization_config = TritonModelOptimizationConfig()
        tensorrt_common_config = TensorRTCommonConfig()
        dynamic_batching_config = TritonDynamicBatchingConfig()
        instances_config = TritonModelInstancesConfig({DeviceKind.GPU: 1})
        backend_parameters_config = TritonCustomBackendParametersConfig()
        initial_model_config_generator = TritonModelConfigGenerator(
            src_model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            tensorrt_common_config=tensorrt_common_config,
            dynamic_batching_config=dynamic_batching_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )
        initial_model_config_generator.save(config_path)

        parsed_model_config_generator = TritonModelConfigGenerator.parse_triton_config_pbtxt(config_path)

        batching_config.max_batch_size = 0

        assert parsed_model_config_generator.batching_config == batching_config
        assert parsed_model_config_generator.model.signature == src_model.signature
        assert parsed_model_config_generator.optimization_config == optimization_config
        assert parsed_model_config_generator.dynamic_batching_config == dynamic_batching_config
        assert parsed_model_config_generator.instances_config == instances_config


@pytest.mark.parametrize(
    "max_batch_size,model_filename,signature",
    [CASE_TENSORRT_PLAN_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES, CASE_TENSORRT_PLAN_IMAGE_MODEL_WITH_DYNAMIC_AXES],
)
def test_model_config_parsing_signature_with_dynamic_batching(monkeypatch, max_batch_size, model_filename, signature):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # create dummy triton model repo structure
        model_path = temp_dir / "1" / model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w"):
            pass

        config_path = temp_dir / "config.pbtxt"

        src_model = Model("dummy", model_path, signature_if_missing=signature)
        batching_config = TritonBatchingConfig(max_batch_size=max_batch_size, batching=Batching.DYNAMIC)
        optimization_config = TritonModelOptimizationConfig()
        tensorrt_common_config = TensorRTCommonConfig()
        dynamic_batching_config = TritonDynamicBatchingConfig()
        instances_config = TritonModelInstancesConfig({DeviceKind.GPU: 1})
        backend_parameters_config = TritonCustomBackendParametersConfig()
        initial_model_config_generator = TritonModelConfigGenerator(
            src_model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            tensorrt_common_config=tensorrt_common_config,
            dynamic_batching_config=dynamic_batching_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )
        initial_model_config_generator.save(config_path)

        parsed_model_config_generator = TritonModelConfigGenerator.parse_triton_config_pbtxt(config_path)

        dynamic_batching_config.preferred_batch_sizes = [max_batch_size]

        assert parsed_model_config_generator.batching_config == batching_config
        assert parsed_model_config_generator.model.signature == src_model.signature
        assert parsed_model_config_generator.optimization_config == optimization_config
        assert parsed_model_config_generator.dynamic_batching_config == dynamic_batching_config
        assert parsed_model_config_generator.instances_config == instances_config


@pytest.mark.parametrize(
    "max_batch_size,model_filename,signature",
    [CASE_TENSORRT_PLAN_SIMPLE_IMAGE_MODEL_WITH_STATIC_AXES, CASE_TENSORRT_PLAN_IMAGE_MODEL_WITH_DYNAMIC_AXES],
)
def test_model_config_parsing_signature_with_dynamic_batching_configured(
    monkeypatch, max_batch_size, model_filename, signature
):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # create dummy triton model repo structure
        model_path = temp_dir / "1" / model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w"):
            pass

        config_path = temp_dir / "config.pbtxt"

        src_model = Model("dummy", model_path, signature_if_missing=signature)
        batching_config = TritonBatchingConfig(max_batch_size=max_batch_size, batching=Batching.DYNAMIC)
        optimization_config = TritonModelOptimizationConfig()
        tensorrt_common_config = TensorRTCommonConfig()
        dynamic_batching_config = TritonDynamicBatchingConfig(preferred_batch_sizes=[1, 2], max_queue_delay_us=100)
        instances_config = TritonModelInstancesConfig({DeviceKind.GPU: 1})
        backend_parameters_config = TritonCustomBackendParametersConfig()
        initial_model_config_generator = TritonModelConfigGenerator(
            src_model,
            batching_config=batching_config,
            optimization_config=optimization_config,
            tensorrt_common_config=tensorrt_common_config,
            dynamic_batching_config=dynamic_batching_config,
            instances_config=instances_config,
            backend_parameters_config=backend_parameters_config,
        )
        initial_model_config_generator.save(config_path)

        parsed_model_config_generator = TritonModelConfigGenerator.parse_triton_config_pbtxt(config_path)

        assert parsed_model_config_generator.batching_config == batching_config
        assert parsed_model_config_generator.model.signature == src_model.signature
        assert parsed_model_config_generator.optimization_config == optimization_config
        assert parsed_model_config_generator.dynamic_batching_config == dynamic_batching_config
        assert parsed_model_config_generator.instances_config == instances_config
