# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import tempfile

import numpy as np
import pytest

from model_navigator.triton.model_config import ModelConfig
from model_navigator.triton.model_config_generator import ModelConfigGenerator
from model_navigator.triton.specialized_configs import (
    AutoMixedPrecisionAccelerator,
    Backend,
    DeviceKind,
    DynamicBatcher,
    GPUIOAccelerator,
    InputTensorFormat,
    InputTensorSpec,
    InstanceGroup,
    ModelWarmup,
    ModelWarmupInput,
    ModelWarmupInputDataType,
    ONNXOptimization,
    OpenVINOAccelerator,
    OutputTensorSpec,
    Platform,
    QueuePolicy,
    SequenceBatcher,
    SequenceBatcherControl,
    SequenceBatcherControlInput,
    SequenceBatcherControlKind,
    SequenceBatcherInitialState,
    SequenceBatcherState,
    SequenceBatcherStrategyDirect,
    SequenceBatcherStrategyOldest,
    TensorFlowOptimization,
    TensorRTAccelerator,
    TensorRTOptimization,
    TensorRTOptPrecision,
    TimeoutAction,
)

full_model_config = ModelConfig(
    model_name="simple",
    backend=Backend.TensorRT,
    batching=True,
    max_batch_size=16,
    batcher=DynamicBatcher(
        preferred_batch_size=[16, 32],
        max_queue_delay_microseconds=100,
        preserve_ordering=True,
        priority_levels=3,
        default_priority_level=1,
        default_queue_policy=QueuePolicy(
            allow_timeout_override=True,
            timeout_action=TimeoutAction.DELAY,
            default_timeout_microseconds=100,
            max_queue_size=2,
        ),
        priority_queue_policy={
            2: QueuePolicy(
                allow_timeout_override=True,
                timeout_action=TimeoutAction.DELAY,
                default_timeout_microseconds=100,
                max_queue_size=3,
            )
        },
    ),
    instance_groups=[
        InstanceGroup(
            kind=DeviceKind.KIND_CPU,
            name="cpu",
            count=1,
            passive=True,
            host_policy="cpu_0",
        ),
        InstanceGroup(
            kind=DeviceKind.KIND_GPU,
            name="gpu",
            count=2,
            gpus=[1, 3],
            host_policy="gpus_1",
            profile=["1", "128"],
        ),
    ],
    parameters={
        "parameter1": "value1",
        "parameter2": "value2",
    },
    inputs=[
        InputTensorSpec(
            name="INPUT_1",
            dtype=np.dtype("float32"),
            shape=(-1,),
            reshape=(1000,),
            is_shape_tensor=True,
            format=InputTensorFormat.FORMAT_NCHW,
            allow_ragged_batch=True,
        ),
        InputTensorSpec(
            name="INPUT_2",
            dtype=np.dtype("bytes"),
            shape=(100, 100),
            reshape=(1000, 1000),
            format=InputTensorFormat.FORMAT_NHWC,
            optional=True,
        ),
        InputTensorSpec(
            name="INPUT_3",
            dtype=np.dtype("int32"),
            shape=(100, 100),
            reshape=(1000, 1000),
            format=InputTensorFormat.FORMAT_NHWC,
            optional=True,
        ),
    ],
    outputs=[
        OutputTensorSpec(
            name="OUTPUT_1",
            dtype=np.dtype("int32"),
            shape=(1000,),
            reshape=(3000,),
            is_shape_tensor=True,
            label_filename="file.txt",
        ),
    ],
    warmup={
        "Warmup1": ModelWarmup(
            inputs={
                "INPUT_1": ModelWarmupInput(
                    dtype=np.dtype("float32"),
                    shape=(-1,),
                    input_data_type=ModelWarmupInputDataType.RANDOM,
                ),
                "INPUT_2": ModelWarmupInput(
                    dtype=np.dtype("bytes"),
                    shape=(100, 100),
                    input_data_type=ModelWarmupInputDataType.ZERO,
                ),
                "INPUT_3": ModelWarmupInput(
                    dtype=np.dtype("int32"),
                    shape=(100, 100),
                    input_data_type=ModelWarmupInputDataType.FILE,
                    input_data_file=pathlib.Path("file.data"),
                ),
            },
        ),
        "Warmup2": ModelWarmup(
            batch_size=2,
            iterations=1,
            inputs={
                "INPUT_1": ModelWarmupInput(
                    dtype=np.dtype("float32"),
                    shape=(-1,),
                    input_data_type=ModelWarmupInputDataType.RANDOM,
                ),
                "INPUT_2": ModelWarmupInput(
                    dtype=np.dtype("bytes"),
                    shape=(100, 100),
                    input_data_type=ModelWarmupInputDataType.ZERO,
                ),
                "INPUT_3": ModelWarmupInput(
                    dtype=np.dtype("int32"),
                    shape=(100, 100),
                    input_data_type=ModelWarmupInputDataType.FILE,
                    input_data_file=pathlib.Path("file.data"),
                ),
            },
        ),
    },
)


def _load_config(config_path: pathlib.Path):
    """Load model config from path.

    Args:
        config_path: path to file with model config

    Returns:
        Dictionary with configuration
    """
    from google.protobuf import json_format, text_format  # pytype: disable=pyi-error
    from tritonclient.grpc import model_config_pb2  # pytype: disable=import-error

    with config_path.open("r") as config_file:
        payload = config_file.read()
        model_config_proto = text_format.Parse(payload, model_config_pb2.ModelConfig())

    model_config_dict = json_format.MessageToDict(model_config_proto, preserving_proto_field_name=True)
    return model_config_dict


def test_get_config_call_config_generator_methods_when_sequence_batcher_used(mocker):
    mock_set_model_signature = mocker.patch.object(ModelConfigGenerator, "_set_model_signature")
    mock_set_instance_group = mocker.patch.object(ModelConfigGenerator, "_set_instance_group")
    mock_set_model_parameters = mocker.patch.object(ModelConfigGenerator, "_set_parameters")
    mock_set_response_cache = mocker.patch.object(ModelConfigGenerator, "_set_response_cache")
    mock_set_sequence_batching = mocker.patch.object(ModelConfigGenerator, "_set_sequence_batching")

    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        batcher=SequenceBatcher(),
    )

    generator = ModelConfigGenerator(model_config)
    model_config_data = generator._get_config()

    assert model_config_data == {
        "name": "simple",
        "backend": "onnxruntime",
        "max_batch_size": 4,
    }

    assert mock_set_model_signature.called is True  # pytype: disable=attribute-error
    assert mock_set_instance_group.called is True  # pytype: disable=attribute-error
    assert mock_set_model_parameters.called is True  # pytype: disable=attribute-error
    assert mock_set_response_cache.called is True  # pytype: disable=attribute-error
    assert mock_set_sequence_batching.called is True  # pytype: disable=attribute-error


def test_get_config_call_config_generator_methods_when_dynamic_batcher_used(mocker):
    mock_set_model_signature = mocker.patch.object(ModelConfigGenerator, "_set_model_signature")
    mock_set_instance_group = mocker.patch.object(ModelConfigGenerator, "_set_instance_group")
    mock_set_model_parameters = mocker.patch.object(ModelConfigGenerator, "_set_parameters")
    mock_set_response_cache = mocker.patch.object(ModelConfigGenerator, "_set_response_cache")
    mock_set_dynamic_batching = mocker.patch.object(ModelConfigGenerator, "_set_dynamic_batching")

    model_config = ModelConfig(model_name="simple", backend=Backend.ONNXRuntime, batcher=DynamicBatcher())

    generator = ModelConfigGenerator(model_config)
    model_config_data = generator._get_config()

    assert model_config_data == {
        "name": "simple",
        "backend": "onnxruntime",
        "max_batch_size": 4,
    }

    assert mock_set_model_signature.called is True  # pytype: disable=attribute-error
    assert mock_set_instance_group.called is True  # pytype: disable=attribute-error
    assert mock_set_model_parameters.called is True  # pytype: disable=attribute-error
    assert mock_set_response_cache.called is True  # pytype: disable=attribute-error
    assert mock_set_dynamic_batching.called is True  # pytype: disable=attribute-error


@pytest.mark.parametrize("backend", list(Backend))
def test_to_file_set_backend_when_provided(backend):
    model_config = ModelConfig(
        model_name="simple",
        backend=backend,
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": backend.value,
            "max_batch_size": 4,
            "dynamic_batching": {},
        }


@pytest.mark.parametrize("platform", list(Platform))
def test_to_file_set_platform_when_provided(platform):
    model_config = ModelConfig(
        model_name="simple",
        platform=platform,
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "platform": platform.value,
            "max_batch_size": 4,
            "dynamic_batching": {},
        }


def test_to_file_set_platform_when_platform_and_backend_provided():
    model_config = ModelConfig(
        model_name="simple",
        platform=Platform.ONNXRuntimeONNX,
        backend=Backend.ONNXRuntime,
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "platform": "onnxruntime_onnx",
            "max_batch_size": 4,
            "dynamic_batching": {},
        }


def test_to_file_set_max_batch_size_to_0_when_batching_is_disabled():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        batching=False,
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        # Missing max_batch_size indicate that value is 0
        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
        }


def test_to_file_set_max_batch_size_and_dynamic_batching_field_when_batching_enabled():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        batching=True,
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
        }


def test_to_file_set_defined_max_batch_size_and_dynamic_batching_field_when_batching_enabled_and_value_passed():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        batching=True,
        max_batch_size=16,
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 16,
            "dynamic_batching": {},
        }


def test_to_file_set_dynamic_batching_config_when_dynamic_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            preferred_batch_size=[16, 32],
            max_queue_delay_microseconds=100,
            preserve_ordering=True,
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 16,
            "dynamic_batching": {
                "preferred_batch_size": [16, 32],
                "max_queue_delay_microseconds": "100",
                "preserve_ordering": True,
            },
        }


def test_to_file_set_dynamic_batching_config_when_default_queue_policy_for_dynamic_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            default_queue_policy=QueuePolicy(
                allow_timeout_override=True,
                timeout_action=TimeoutAction.DELAY,
                default_timeout_microseconds=100,
                max_queue_size=2,
            )
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 16,
            "dynamic_batching": {
                "default_queue_policy": {
                    "allow_timeout_override": 1,
                    "timeout_action": "DELAY",
                    "default_timeout_microseconds": "100",
                    "max_queue_size": 2,
                }
            },
        }


def test_to_file_set_dynamic_batching_config_when_priority_queue_policy_for_dynamic_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        batching=True,
        backend=Backend.ONNXRuntime,
        max_batch_size=16,
        batcher=DynamicBatcher(
            priority_levels=3,
            default_priority_level=1,
            priority_queue_policy={
                2: QueuePolicy(
                    allow_timeout_override=True,
                    timeout_action=TimeoutAction.DELAY,
                    default_timeout_microseconds=100,
                    max_queue_size=2,
                )
            },
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 16,
            "dynamic_batching": {
                "priority_levels": "3",
                "default_priority_level": "1",
                "priority_queue_policy": {
                    "2": {
                        "allow_timeout_override": 1,
                        "timeout_action": "DELAY",
                        "default_timeout_microseconds": "100",
                        "max_queue_size": 2,
                    }
                },
            },
        }


def test_to_file_set_sequence_batching_when_sequence_batcher_used():
    model_config = ModelConfig(model_name="simple", backend=Backend.TensorRT, batcher=SequenceBatcher())
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {},
        }


def test_to_file_set_sequence_batching_when_strategy_oldest_with_minimal_data_for_sequence_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        batcher=SequenceBatcher(
            strategy=SequenceBatcherStrategyOldest(
                max_candidate_sequences=10,
            )
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {
                "oldest": {
                    "max_candidate_sequences": 10,
                }
            },
        }


def test_to_file_set_sequence_batching_when_strategy_oldest_with_full_data_for_sequence_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        batcher=SequenceBatcher(
            strategy=SequenceBatcherStrategyOldest(
                max_candidate_sequences=10,
                preferred_batch_size=[1, 4, 8],
                max_queue_delay_microseconds=100,
            )
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {
                "oldest": {
                    "max_candidate_sequences": 10,
                    "preferred_batch_size": [1, 4, 8],
                    "max_queue_delay_microseconds": "100",
                }
            },
        }


def test_to_file_set_sequence_batching_when_strategy_direct_with_minimal_data_for_sequence_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        batcher=SequenceBatcher(strategy=SequenceBatcherStrategyDirect()),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {"direct": {}},
        }


def test_to_file_set_sequence_batching_when_strategy_direct_with_full_data_for_sequence_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        batcher=SequenceBatcher(
            strategy=SequenceBatcherStrategyDirect(
                max_queue_delay_microseconds=100,
                minimum_slot_utilization=10,
            )
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {
                "direct": {
                    "max_queue_delay_microseconds": "100",
                    "minimum_slot_utilization": 10,
                }
            },
        }


def test_to_file_set_sequence_batching_when_control_input_provided_for_sequence_batcher_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        batcher=SequenceBatcher(
            control_inputs=[
                SequenceBatcherControlInput(
                    input_name="input_0",
                    controls=[
                        SequenceBatcherControl(
                            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID,
                            dtype=np.float32,
                        ),
                        SequenceBatcherControl(
                            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_START,
                            fp32_false_true=[0, 1],
                        ),
                        SequenceBatcherControl(
                            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_END,
                            int32_false_true=[0, 2],
                        ),
                        SequenceBatcherControl(
                            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_READY,
                            bool_false_true=[False, True],
                        ),
                    ],
                )
            ]
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {
                "control_input": [
                    {
                        "name": "input_0",
                        "control": [
                            {
                                "kind": "CONTROL_SEQUENCE_CORRID",
                                "data_type": "TYPE_FP32",
                            },
                            {
                                # "kind": "CONTROL_SEQUENCE_START", - default 0 value
                                "fp32_false_true": [0.0, 1.0],
                            },
                            {
                                "kind": "CONTROL_SEQUENCE_END",
                                "int32_false_true": [0, 2],
                            },
                            {
                                "kind": "CONTROL_SEQUENCE_READY",
                                "bool_false_true": [False, True],
                            },
                        ],
                    }
                ]
            },
        }


def test_to_file_set_sequence_batching_when_state_provided_for_sequence_batcher_with_zero_data_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        batcher=SequenceBatcher(
            states=[
                SequenceBatcherState(
                    input_name="input_1",
                    output_name="output_1",
                    dtype=np.float32,
                    shape=(-1,),
                ),
                SequenceBatcherState(
                    input_name="input_2",
                    output_name="output_2",
                    dtype=np.int32,
                    shape=(-1, -1),
                    initial_states=[
                        SequenceBatcherInitialState(
                            name="initialization",
                            dtype=np.int32,
                            shape=(-1, -1),
                            zero_data=True,
                        )
                    ],
                ),
            ]
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {
                "state": [
                    {
                        "input_name": "input_1",
                        "output_name": "output_1",
                        "data_type": "TYPE_FP32",
                        "dims": ["-1"],
                    },
                    {
                        "input_name": "input_2",
                        "output_name": "output_2",
                        "data_type": "TYPE_INT32",
                        "dims": ["-1", "-1"],
                        "initial_state": [
                            {
                                "name": "initialization",
                                "data_type": "TYPE_INT32",
                                "dims": ["-1", "-1"],
                                "zero_data": True,
                            }
                        ],
                    },
                ]
            },
        }


def test_to_file_set_sequence_batching_when_state_provided_for_sequence_batcher_with_data_file_used():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        batcher=SequenceBatcher(
            states=[
                SequenceBatcherState(
                    input_name="input_1",
                    output_name="output_1",
                    dtype=np.float32,
                    shape=(-1,),
                ),
                SequenceBatcherState(
                    input_name="input_2",
                    output_name="output_2",
                    dtype=np.int32,
                    shape=(-1, -1),
                    initial_states=[
                        SequenceBatcherInitialState(
                            name="initialization",
                            dtype=np.int32,
                            shape=(-1, -1),
                            data_file=pathlib.Path("file.data"),
                        )
                    ],
                ),
            ]
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "sequence_batching": {
                "state": [
                    {
                        "input_name": "input_1",
                        "output_name": "output_1",
                        "data_type": "TYPE_FP32",
                        "dims": ["-1"],
                    },
                    {
                        "input_name": "input_2",
                        "output_name": "output_2",
                        "data_type": "TYPE_INT32",
                        "dims": ["-1", "-1"],
                        "initial_state": [
                            {
                                "name": "initialization",
                                "data_type": "TYPE_INT32",
                                "dims": ["-1", "-1"],
                                "data_file": "file.data",
                            }
                        ],
                    },
                ]
            },
        }


def test_to_file_set_instance_configuration_when_single_config_provided():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        instance_groups=[
            InstanceGroup(kind=DeviceKind.KIND_GPU),
        ],
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "instance_group": [
                {"kind": DeviceKind.KIND_GPU.value},
            ],
        }


def test_to_file_set_instance_configuration_when_single_multiple_configs_provided():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        instance_groups=[
            InstanceGroup(
                name="GPU",
                kind=DeviceKind.KIND_GPU,
                gpus=[0, 1, 2],
                host_policy="gpus_0",
            ),
            InstanceGroup(
                name="CPU",
                kind=DeviceKind.KIND_CPU,
                count=10,
                passive=False,
            ),
            InstanceGroup(
                name="AUTO",
                kind=DeviceKind.KIND_AUTO,
            ),
        ],
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "instance_group": [
                {"name": "GPU", "kind": DeviceKind.KIND_GPU.value, "gpus": [0, 1, 2], "host_policy": "gpus_0"},
                {"name": "CPU", "kind": DeviceKind.KIND_CPU.value, "count": 10},
                {"name": "AUTO"},
            ],
        }


def test_to_file_set_parameters_configuration_when_parameters_provided():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        parameters={
            "parameter1": "value1",
            "parameter2": "value2",
        },
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "parameters": {
                "parameter1": {"string_value": "value1"},
                "parameter2": {"string_value": "value2"},
            },
        }


def test_to_file_set_inputs_and_outputs_when_minimal_specification_provided():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        inputs=[
            InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(-1,)),
            InputTensorSpec(name="INPUT_2", dtype=np.dtype("bytes"), shape=(-1,)),
        ],
        outputs=[
            OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
        ],
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "input": [
                {"name": "INPUT_1", "data_type": "TYPE_FP32", "dims": ["-1"]},
                {"name": "INPUT_2", "data_type": "TYPE_STRING", "dims": ["-1"]},
            ],
            "output": [
                {"name": "OUTPUT_1", "data_type": "TYPE_INT32", "dims": ["1000"]},
            ],
        }


def test_to_file_set_inputs_and_outputs_when_type_provided_provided():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        inputs=[
            InputTensorSpec(name="INPUT_1", dtype=np.float32, shape=(-1,)),
            InputTensorSpec(name="INPUT_2", dtype=np.bytes_, shape=(-1,)),
        ],
        outputs=[
            OutputTensorSpec(name="OUTPUT_1", dtype=np.int32, shape=(1000,)),
        ],
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "input": [
                {"name": "INPUT_1", "data_type": "TYPE_FP32", "dims": ["-1"]},
                {"name": "INPUT_2", "data_type": "TYPE_STRING", "dims": ["-1"]},
            ],
            "output": [
                {"name": "OUTPUT_1", "data_type": "TYPE_INT32", "dims": ["1000"]},
            ],
        }


def test_to_file_set_inputs_and_outputs_when_full_specification_provided():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        inputs=[
            InputTensorSpec(
                name="INPUT_1",
                dtype=np.dtype("float32"),
                shape=(-1,),
                allow_ragged_batch=True,
                is_shape_tensor=True,
                optional=True,
                format=InputTensorFormat.FORMAT_NCHW,
            ),
            InputTensorSpec(name="INPUT_2", dtype=np.dtype("bytes"), shape=(-1,), reshape=(100,)),
        ],
        outputs=[
            OutputTensorSpec(
                name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,), is_shape_tensor=True, reshape=(500,)
            ),
        ],
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "input": [
                {
                    "name": "INPUT_1",
                    "data_type": "TYPE_FP32",
                    "dims": ["-1"],
                    "allow_ragged_batch": True,
                    "is_shape_tensor": True,
                    "optional": True,
                    "format": "FORMAT_NCHW",
                },
                {
                    "name": "INPUT_2",
                    "data_type": "TYPE_STRING",
                    "dims": ["-1"],
                    "reshape": {"shape": ["100"]},
                },
            ],
            "output": [
                {
                    "name": "OUTPUT_1",
                    "data_type": "TYPE_INT32",
                    "dims": ["1000"],
                    "is_shape_tensor": True,
                    "reshape": {"shape": ["500"]},
                },
            ],
        }


def test_to_file_set_response_cache_when_enabled():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        response_cache=True,
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "response_cache": {"enable": True},
        }


def test_to_file_set_tensorrt_optimization_for_tensorrt_when_provided_for_model():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorRT,
        optimization=TensorRTOptimization(
            cuda_graphs=True,
            eager_batching=True,
            gather_kernel_buffer_threshold=10,
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "optimization": {
                "cuda": {
                    "graphs": True,
                },
                "eager_batching": True,
                "gather_kernel_buffer_threshold": 10,
            },
        }


def test_to_file_set_tensorrt_accelerator_for_onnx_when_provided_for_model():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        optimization=ONNXOptimization(
            accelerator=TensorRTAccelerator(
                precision=TensorRTOptPrecision.FP16,
                max_workspace_size=8192,  # 8GB
                max_cached_engines=4,
                minimum_segment_size=1,
            )
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "optimization": {
                "execution_accelerators": {
                    "gpu_execution_accelerator": [
                        {
                            "name": "tensorrt",
                            "parameters": {
                                "precision_mode": "FP16",
                                "max_workspace_size_bytes": "8589934592",
                                "max_cached_engines": "4",
                                "minimum_segment_size": "1",
                            },
                        }
                    ],
                },
            },
        }


def test_to_file_set_openvino_accelerator_for_onnx_when_provided_for_model():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.ONNXRuntime,
        optimization=ONNXOptimization(
            accelerator=OpenVINOAccelerator(),
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "onnxruntime",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "optimization": {
                "execution_accelerators": {
                    "cpu_execution_accelerator": [
                        {
                            "name": "openvino",
                        }
                    ],
                },
            },
        }


def test_to_file_set_tenssort_accelerator_for_tensorflow_when_provided_for_model():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorFlow,
        optimization=TensorFlowOptimization(
            accelerator=TensorRTAccelerator(
                precision=TensorRTOptPrecision.FP16,
                max_workspace_size=8192,  # 8GB
                max_cached_engines=4,
                minimum_segment_size=1,
            )
        ),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorflow",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "optimization": {
                "execution_accelerators": {
                    "gpu_execution_accelerator": [
                        {
                            "name": "tensorrt",
                            "parameters": {
                                "precision_mode": "FP16",
                                "max_workspace_size_bytes": "8589934592",
                                "max_cached_engines": "4",
                                "minimum_segment_size": "1",
                            },
                        }
                    ],
                },
            },
        }


def test_to_file_set_amp_accelerator_for_tensorflow_when_provided_for_model():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorFlow,
        optimization=TensorFlowOptimization(accelerator=AutoMixedPrecisionAccelerator()),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorflow",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "optimization": {
                "execution_accelerators": {
                    "gpu_execution_accelerator": [
                        {
                            "name": "auto_mixed_precision",
                        }
                    ],
                },
            },
        }


def test_to_file_set_gpuio_accelerator_for_tensorflow_when_provided_for_model():
    model_config = ModelConfig(
        model_name="simple",
        backend=Backend.TensorFlow,
        optimization=TensorFlowOptimization(accelerator=GPUIOAccelerator()),
    )
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorflow",
            "max_batch_size": 4,
            "dynamic_batching": {},
            "optimization": {
                "execution_accelerators": {
                    "gpu_execution_accelerator": [
                        {
                            "name": "gpu_io",
                        }
                    ],
                },
            },
        }


def test_to_file_save_config_to_file_when_full_config_specified():
    generator = ModelConfigGenerator(full_model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "tensorrt",
            "max_batch_size": 16,
            "dynamic_batching": {
                "preferred_batch_size": [16, 32],
                "max_queue_delay_microseconds": "100",
                "preserve_ordering": True,
                "priority_levels": "3",
                "default_priority_level": "1",
                "default_queue_policy": {
                    "allow_timeout_override": True,
                    "timeout_action": "DELAY",
                    "default_timeout_microseconds": "100",
                    "max_queue_size": 2,
                },
                "priority_queue_policy": {
                    "2": {
                        "allow_timeout_override": True,
                        "timeout_action": "DELAY",
                        "default_timeout_microseconds": "100",
                        "max_queue_size": 3,
                    }
                },
            },
            "instance_group": [
                {
                    "name": "cpu",
                    "count": 1,
                    "kind": "KIND_CPU",
                    "passive": True,
                    "host_policy": "cpu_0",
                },
                {
                    "name": "gpu",
                    "count": 2,
                    "kind": "KIND_GPU",
                    "gpus": [1, 3],
                    "host_policy": "gpus_1",
                    "profile": ["1", "128"],
                },
            ],
            "input": [
                {
                    "name": "INPUT_1",
                    "data_type": "TYPE_FP32",
                    "dims": ["-1"],
                    "allow_ragged_batch": True,
                    "format": "FORMAT_NCHW",
                    "is_shape_tensor": True,
                    "reshape": {"shape": ["1000"]},
                },
                {
                    "name": "INPUT_2",
                    "data_type": "TYPE_STRING",
                    "dims": ["100", "100"],
                    "format": "FORMAT_NHWC",
                    "reshape": {"shape": ["1000", "1000"]},
                    "optional": True,
                },
                {
                    "name": "INPUT_3",
                    "data_type": "TYPE_INT32",
                    "dims": ["100", "100"],
                    "format": "FORMAT_NHWC",
                    "reshape": {"shape": ["1000", "1000"]},
                    "optional": True,
                },
            ],
            "output": [
                {
                    "name": "OUTPUT_1",
                    "data_type": "TYPE_INT32",
                    "dims": ["1000"],
                    "reshape": {"shape": ["3000"]},
                    "label_filename": "file.txt",
                    "is_shape_tensor": True,
                },
            ],
            "parameters": {
                "parameter1": {"string_value": "value1"},
                "parameter2": {"string_value": "value2"},
            },
            "model_warmup": [
                {
                    "name": "Warmup1",
                    "batch_size": 1,
                    "inputs": {
                        "INPUT_1": {
                            "data_type": "TYPE_FP32",
                            "dims": ["-1"],
                            "random_data": True,
                        },
                        "INPUT_2": {
                            "data_type": "TYPE_STRING",
                            "dims": ["100", "100"],
                            "zero_data": True,
                        },
                        "INPUT_3": {
                            "data_type": "TYPE_INT32",
                            "dims": ["100", "100"],
                            "input_data_file": "file.data",
                        },
                    },
                },
                {
                    "name": "Warmup2",
                    "batch_size": 2,
                    "count": 1,
                    "inputs": {
                        "INPUT_1": {
                            "data_type": "TYPE_FP32",
                            "dims": ["-1"],
                            "random_data": True,
                        },
                        "INPUT_2": {
                            "data_type": "TYPE_STRING",
                            "dims": ["100", "100"],
                            "zero_data": True,
                        },
                        "INPUT_3": {
                            "data_type": "TYPE_INT32",
                            "dims": ["100", "100"],
                            "input_data_file": "file.data",
                        },
                    },
                },
            ],
        }
