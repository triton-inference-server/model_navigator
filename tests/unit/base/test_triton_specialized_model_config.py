# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
from typing import Type

import numpy as np
import pytest

from model_navigator.exceptions import ModelNavigatorWrongParameterError
from model_navigator.triton import DecodingMode, KVCacheConfig, PeftCacheConfig, SequenceBatcherStrategyOldest
from model_navigator.triton.specialized_configs import (
    DeviceKind,
    DynamicBatcher,
    InputTensorSpec,
    InstanceGroup,
    ModelWarmup,
    ModelWarmupInput,
    ModelWarmupInputDataType,
    ONNXModelConfig,
    ONNXOptimization,
    OutputTensorSpec,
    PythonModelConfig,
    PyTorchModelConfig,
    QueuePolicy,
    SequenceBatcher,
    SequenceBatcherControl,
    SequenceBatcherControlKind,
    SequenceBatcherInitialState,
    TensorFlowModelConfig,
    TensorFlowOptimization,
    TensorRTLLMModelConfig,
    TensorRTModelConfig,
    TensorRTOptimization,
    TimeoutAction,
)

SPECIALIZED_CONFIGS = (
    ONNXModelConfig,
    PythonModelConfig,
    PyTorchModelConfig,
    TensorFlowModelConfig,
    TensorRTModelConfig,
    TensorRTLLMModelConfig,
)

PLATFORM_CONFIGS = (
    ONNXModelConfig,
    PyTorchModelConfig,
    TensorFlowModelConfig,
    TensorRTModelConfig,
)

INPUTS_CONFIGS = (
    PythonModelConfig,
    PyTorchModelConfig,
)

OPTIMIZATION_CONFIGS = (
    ONNXModelConfig,
    TensorFlowModelConfig,
    TensorRTModelConfig,
)


def _extra_args(clazz):
    extra_args = {}
    if clazz in INPUTS_CONFIGS:
        extra_args = {
            "inputs": [
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(1000,)),
            ],
            "outputs": [
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        }
    return extra_args


@pytest.mark.parametrize("clazz", list(PLATFORM_CONFIGS))
def test_model_config_raise_error_when_unsupported_platform_provided(clazz: Type):
    with pytest.raises(ModelNavigatorWrongParameterError, match="Unsupported platform provided. Use"):
        clazz(
            platform=object(),
            **_extra_args(clazz),
        )


@pytest.mark.parametrize("clazz", list(SPECIALIZED_CONFIGS))
def test_specialized_model_config_raise_error_when_max_batch_size_is_0_and_batching_is_not_disabled(clazz: Type):
    with pytest.raises(ModelNavigatorWrongParameterError, match="The `max_batch_size` must be greater or equal to 1."):
        clazz(
            batching=True,
            max_batch_size=0,
            **_extra_args(clazz),
        )


@pytest.mark.parametrize("clazz", list(SPECIALIZED_CONFIGS))
def test_model_config_raise_error_when_max_batch_size_is_less_then_0_and_batching_is_not_disabled(clazz: Type):
    with pytest.raises(ModelNavigatorWrongParameterError, match="The `max_batch_size` must be greater or equal to 1."):
        clazz(
            batching=True,
            max_batch_size=-1,
            **_extra_args(clazz),
        )


@pytest.mark.parametrize("clazz", list(OPTIMIZATION_CONFIGS))
def test_specialized_model_config_raise_error_when_unsupported_optimization_passed(clazz: Type):
    with pytest.raises(ModelNavigatorWrongParameterError, match="Unsupported optimization type provided."):
        clazz(
            optimization=object(),  # pytype: disable=wrong-arg-types
        )


@pytest.mark.parametrize("clazz", list(SPECIALIZED_CONFIGS))
def test_specialized_model_config_raise_error_when_unsupported_batcher_passed(clazz: Type):
    with pytest.raises(ModelNavigatorWrongParameterError, match="Unsupported batcher type provided."):
        clazz(batcher=object(), **_extra_args(clazz))  # pytype: disable=wrong-arg-types


@pytest.mark.parametrize("clazz", list(INPUTS_CONFIGS))
def test_specialized_model_config_raise_error_when_invalid_input_shape_type_passed(clazz: Type):
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`shape` argument should be <class 'tuple'>, but got <class 'list'>.",
    ):
        clazz(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=[]),  # pytype: disable=wrong-arg-types
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        )


@pytest.mark.parametrize("clazz", list(INPUTS_CONFIGS))
def test_specialized_model_config_raise_error_when_invalid_output_shape_type_passed(clazz: Type):
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`shape` argument should be <class 'tuple'>, but got <class 'list'>.",
    ):
        clazz(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(1000,)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=[]),  # pytype: disable=wrong-arg-types
            ],
        )


@pytest.mark.parametrize("clazz", list(INPUTS_CONFIGS))
def test_specialized_model_config_raise_error_when_empty_input_shapes_passed(clazz: Type):
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Empty shape is not supported.",
    ):
        clazz(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=()),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        )


@pytest.mark.parametrize("clazz", list(INPUTS_CONFIGS))
def test_model_config_raise_error_when_empty_output_shapes_passed(clazz: Type):
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Empty shape is not supported.",
    ):
        clazz(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(1000,)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=()),
            ],
        )


@pytest.mark.parametrize("clazz", list(INPUTS_CONFIGS))
def test_specialized_model_config_raise_error_when_invalid_input_shapes_passed(clazz: Type):
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match=r"shape items should be integers equal to -1 or positive numbers. Got \(-2,\).",
    ):
        clazz(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(-2,)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(1000,)),
            ],
        )


@pytest.mark.parametrize("clazz", list(INPUTS_CONFIGS))
def test_specialized_model_config_raise_error_when_invalid_output_shapes_passed(clazz: Type):
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match=r"shape items should be integers equal to -1 or positive numbers. Got \(0,\).",
    ):
        clazz(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(1000,)),
            ],
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("int32"), shape=(0,)),
            ],
        )


@pytest.mark.parametrize("clazz", list(SPECIALIZED_CONFIGS))
def test_specialized_model_config_raise_error_when_profile_in_instance_group_none_tensorrt_backend(clazz: Type):
    if clazz == TensorRTModelConfig:
        return

    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Invalid `profile` option. The value can be set only for `backend=Backend.TensorRT`",
    ):
        clazz(
            **_extra_args(clazz),
            instance_groups=[
                InstanceGroup(
                    profile=["profile1"],
                ),
            ],
        )


def test_dynamic_batcher_raise_exception_when_invalid_default_priority_level_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError, match="The `default_priority_level` must be between 1 and 5."
    ):
        DynamicBatcher(
            priority_levels=5,
            default_priority_level=6,
        )


def test_dynamic_batcher_raise_exception_when_priority_queue_policy_passed_but_no_default_priority_level():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Provide the `priority_levels` if you want to define `priority_queue_policy` for Dynamic Batching.",
    ):
        DynamicBatcher(
            priority_queue_policy={
                1: QueuePolicy(
                    allow_timeout_override=True,
                    timeout_action=TimeoutAction.DELAY,
                    default_timeout_microseconds=100,
                    max_queue_size=2,
                )
            },
        )


def test_dynamic_batcher_raise_exception_when_invalid_priority_queue_policy_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError, match="Invalid `priority`=6 provided. The value must be between 1 and 5."
    ):
        DynamicBatcher(
            priority_levels=5,
            default_priority_level=2,
            priority_queue_policy={
                6: QueuePolicy(
                    allow_timeout_override=True,
                    timeout_action=TimeoutAction.DELAY,
                    default_timeout_microseconds=100,
                    max_queue_size=2,
                )
            },
        )


def test_instance_group_raise_error_when_count_less_than_1():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The `count` must be greater or equal 1.",
    ):
        InstanceGroup(
            count=0,
        )


def test_instance_group_raise_error_when_unsupported_gpus_option_in_instance_configuration_provided():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`gpus` cannot be set when device is not DeviceKind.KIND_GPU or DeviceKind.KIND_AUTO",
    ):
        InstanceGroup(
            name="CPU",
            kind=DeviceKind.KIND_CPU,
            gpus=[0, 1, 2],
            host_policy="gpus_0",
        )


def test_sequence_batcher_raise_error_when_unsupported_strategy_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Unsupported strategy type provided.",
    ):
        SequenceBatcher(
            strategy=object(),  # pytype: disable=wrong-arg-types
        )


def test_sequence_batcher_control_raise_error_when_corrid_and_no_data_type():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID control type requires `dtype` to be specified.",
    ):
        SequenceBatcherControl(
            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID,
        )


def test_sequence_batcher_control_raise_error_when_corrid_and_unsupported_int32_false_true_provided():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID control type requires `dtype` "
        "to be specified only.",
    ):
        SequenceBatcherControl(
            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID,
            dtype=np.float32,
            int32_false_true=[0, 1],
        )


def test_sequence_batcher_control_raise_error_when_corrid_and_unsupported_fp32_false_true_provided():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID control type requires `dtype` "
        "to be specified only.",
    ):
        SequenceBatcherControl(
            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID,
            dtype=np.float32,
            fp32_false_true=[0, 1],
        )


def test_sequence_batcher_control_raise_error_when_corrid_and_unsupported_bool_false_true_provided():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID control type requires `dtype` "
        "to be specified only.",
    ):
        SequenceBatcherControl(
            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID,
            dtype=np.float32,
            fp32_false_true=[False, True],
        )


@pytest.mark.parametrize(
    "control_type",
    [
        SequenceBatcherControlKind.CONTROL_SEQUENCE_START,
        SequenceBatcherControlKind.CONTROL_SEQUENCE_END,
        SequenceBatcherControlKind.CONTROL_SEQUENCE_READY,
    ],
)
def test_sequence_batcher_control_raise_error_when_start_end_ready_and_datatype_provided(control_type):
    with pytest.raises(
        ModelNavigatorWrongParameterError, match=f"The {control_type} control does not support `dtype` parameter."
    ):
        SequenceBatcherControl(
            kind=control_type,
            dtype=np.float32,
        )


@pytest.mark.parametrize(
    "control_type",
    [
        SequenceBatcherControlKind.CONTROL_SEQUENCE_START,
        SequenceBatcherControlKind.CONTROL_SEQUENCE_END,
        SequenceBatcherControlKind.CONTROL_SEQUENCE_READY,
    ],
)
def test_sequence_batcher_control_raise_error_when_start_end_ready_and_missing_required_parameters(
    control_type,
):
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match=f"The {control_type} control type requires one of: "
        "`int32_false_true`, `fp32_false_true`, `bool_false_true` to be specified.",
    ):
        SequenceBatcherControl(
            kind=control_type,
        )


def test_sequence_batcher_control_raise_error_when_invalid_int32_false_true_parameters():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The `int32_false_true` field should be two element list with false and true values. "
        r"Example: \[0 , 1\]",
    ):
        SequenceBatcherControl(
            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_START,
            int32_false_true=[1],
        )


def test_sequence_batcher_control_raise_error_when_invalid_fp32_false_true_parameters():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The `fp32_false_true` field should be two element list with false and true values. "
        r"Example: \[0 , 1\]",
    ):
        SequenceBatcherControl(
            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_START,
            fp32_false_true=[1],
        )


def test_sequence_batcher_control_raise_error_when_invalid_bool_false_true_parameters():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="The `bool_false_true` field should be two element list with false and true values. "
        r"Example: \[False, True\]",
    ):
        SequenceBatcherControl(
            kind=SequenceBatcherControlKind.CONTROL_SEQUENCE_START,
            bool_false_true=[False],
        )


def test_sequence_batcher_initial_state_raise_error_when_miss_parameters():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="zero_data or data_file has to be defined. None was provided.",
    ):
        SequenceBatcherInitialState(
            name="input",
            dtype=np.float32,
            shape=(-1,),
        )


def test_sequence_batcher_initial_state_raise_error_when_have_to_many_parameters():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="zero_data or data_file has to be defined. Both were provided.",
    ):
        SequenceBatcherInitialState(
            name="input",
            dtype=np.float32,
            shape=(-1,),
            zero_data=True,
            data_file=pathlib.Path("file.data"),
        )


def test_onnx_optimization_raise_error_when_invalid_accelerator_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Unsupported accelerator type provided.",
    ):
        ONNXOptimization(accelerator=object())  # pytype: disable=wrong-arg-types


def test_tensorflow_optimization_raise_error_when_invalid_accelerator_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Unsupported accelerator type provided.",
    ):
        TensorFlowOptimization(accelerator=object())  # pytype: disable=wrong-arg-types


def test_tensorrt_optimization_raise_error_when_none_of_accelerators_enabled():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="At least one of the optimization options should be enabled.",
    ):
        TensorRTOptimization()


def test_model_warmup_raise_error_when_batch_size_smaller_than_1():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`batch_size` must be greater or equal 1.",
    ):
        ModelWarmup(
            batch_size=0,
            inputs={
                "INPUT_1": ModelWarmupInput(
                    dtype=np.dtype("float32"),
                    shape=(1000,),
                    input_data_type=ModelWarmupInputDataType.RANDOM,
                )
            },
        )


def test_model_warmup_raise_error_when_iterations_smaller_than_1():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="iterations` must be greater or equal 0.",
    ):
        ModelWarmup(
            inputs={
                "INPUT_1": ModelWarmupInput(
                    dtype=np.dtype("float32"),
                    shape=(1000,),
                    input_data_type=ModelWarmupInputDataType.RANDOM,
                )
            },
            iterations=-1,
        )


def test_model_warmup_raise_error_when_file_input_data_and_file_not_provided():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`input_data_file` is required. Set the file path.",
    ):
        ModelWarmup(
            inputs={
                "INPUT_1": ModelWarmupInput(
                    dtype=np.dtype("float32"),
                    shape=(1000,),
                    input_data_type=ModelWarmupInputDataType.FILE,
                )
            },
        )


def test_model_warmup_raise_error_when_random_input_data_and_file_provided():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`input_data_file` is not required. Remove the parameter.",
    ):
        ModelWarmup(
            inputs={
                "INPUT_1": ModelWarmupInput(
                    dtype=np.dtype("float32"),
                    shape=(1000,),
                    input_data_type=ModelWarmupInputDataType.RANDOM,
                    input_data_file=pathlib.Path("my-file.data"),
                )
            },
        )


def test_kv_cache_config_raise_error_when_invalid_max_tokens():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`max_tokens` must be greater than 0.",
    ):
        KVCacheConfig(max_tokens=0)


def test_kv_cache_config_raise_error_when_invalid_sink_token_length():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`sink_token_length` must be greater than 0.",
    ):
        KVCacheConfig(sink_token_length=0)


def test_kv_cache_config_raise_error_when_invalid_max_attention_window():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`max_attention_window` must be greater than 0.",
    ):
        KVCacheConfig(max_attention_window=0)


def test_kv_cache_config_raise_error_when_invalid_free_gpu_memory_fraction():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`free_gpu_memory_fraction` must be between 0.0 and 1.0.",
    ):
        KVCacheConfig(free_gpu_memory_fraction=-0.1)

    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`free_gpu_memory_fraction` must be between 0.0 and 1.0.",
    ):
        KVCacheConfig(free_gpu_memory_fraction=1.1)


def test_kv_cache_config_raise_error_when_invalid_host_cache_size():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`host_cache_size` must be greater than 0.",
    ):
        KVCacheConfig(host_cache_size=0)


def test_kv_cache_config_raise_error_when_invalid_onboard_blocks():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`onboard_blocks` must be greater than 0.",
    ):
        KVCacheConfig(onboard_blocks=0)


def test_peft_cache_config_raise_error_when_invalid_optimal_adapter_size():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`optimal_adapter_size` must be greater than 0.",
    ):
        PeftCacheConfig(optimal_adapter_size=0)


def test_peft_cache_config_raise_error_when_invalid_max_adapter_size():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`max_adapter_size` must be greater than 0.",
    ):
        PeftCacheConfig(max_adapter_size=0)


def test_peft_cache_config_raise_error_when_max_adapter_size_lower_than_optimal_adapter_size():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`max_adapter_size` must be greater than or equal to `optimal_adapter_size`.",
    ):
        PeftCacheConfig(max_adapter_size=10, optimal_adapter_size=20)


def test_peft_cache_config_raise_error_when_invalid_gpu_memory_fraction():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`gpu_memory_fraction` must be between 0.0 and 1.0.",
    ):
        PeftCacheConfig(gpu_memory_fraction=-0.1)

    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`gpu_memory_fraction` must be between 0.0 and 1.0.",
    ):
        PeftCacheConfig(gpu_memory_fraction=1.1)


def test_peft_cache_config_raise_error_when_invalid_host_memory_bytes():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`host_memory_bytes` must be greater than 0.",
    ):
        PeftCacheConfig(host_memory_bytes=0)


def test_tensorrt_llm_model_config_raise_error_when_instance_groups_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Instance groups are not supported for TensorRT-LLM backend has predefined instance groups.",
    ):
        TensorRTLLMModelConfig(
            instance_groups=[
                InstanceGroup(
                    kind=DeviceKind.KIND_GPU,
                    name="gpu",
                    count=2,
                ),
            ]
        )


def test_tensorrt_llm_model_config_raise_error_when_inputs_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Inputs parameters is not supported as TensorRT-LLM backend has predefined shapes.",
    ):
        TensorRTLLMModelConfig(
            inputs=[
                InputTensorSpec(name="INPUT_1", dtype=np.dtype("float32"), shape=(-1,)),
            ],
        )


def test_tensorrt_llm_model_config_raise_error_when_outputs_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Outputs parameters is not supported as TensorRT-LLM backend has predefined shapes.",
    ):
        TensorRTLLMModelConfig(
            outputs=[
                OutputTensorSpec(name="OUTPUT_1", dtype=np.dtype("float32"), shape=(-1,)),
            ],
        )


def test_tensorrt_llm_model_config_raise_error_when_default_model_filename_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Default model filename is not supported.",
    ):
        TensorRTLLMModelConfig(default_model_filename="custom_name")


def test_tensorrt_llm_model_config_raise_error_when_non_dynamic_batcher_provided():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Batcher must be an instance of DynamicBatcher.",
    ):
        TensorRTLLMModelConfig(
            batcher=SequenceBatcher(
                strategy=SequenceBatcherStrategyOldest(
                    max_candidate_sequences=10,
                )
            )
        )


def test_tensorrt_llm_model_config_raise_error_when_decoding_mode_is_medusa_but_medusa_choices_not_passed():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="Medusa choices are required for Medusa decoding mode.",
    ):
        TensorRTLLMModelConfig(
            decoding_mode=DecodingMode.MEDUSA,
        )


def test_tensorrt_llm_model_config_raise_error_when_invalid_medusa_choices():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="""Medusa choices item must be a list or tuple. See an example at: """
        """https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py""",
    ):
        TensorRTLLMModelConfig(
            medusa_choices=[{"1": 1, "2": 2}],  # pytype: disable=wrong-arg-types
        )

    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="""Medusa choices must be a list or tuple. See an example at: """
        """https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py""",
    ):
        TensorRTLLMModelConfig(
            medusa_choices={"1": [1, 2, 3], "2": [1, 2]},  # pytype: disable=wrong-arg-types
        )


def test_tensorrt_llm_model_config_raise_error_when_invalid_gpu_weights_percent():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`gpu_weights_percent` must be between 0.0 and 1.0.",
    ):
        TensorRTLLMModelConfig(gpu_weights_percent=-0.1)

    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`gpu_weights_percent` must be between 0.0 and 1.0.",
    ):
        TensorRTLLMModelConfig(gpu_weights_percent=1.1)


def test_tensorrt_llm_model_config_raise_error_when_invalid_cancellation_check_period_ms():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`cancellation_check_period_ms` must be greater than 0.",
    ):
        TensorRTLLMModelConfig(cancellation_check_period_ms=0)


def test_tensorrt_llm_model_config_raise_error_when_invalid_stats_check_period_ms():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`stats_check_period_ms` must be greater than 0.",
    ):
        TensorRTLLMModelConfig(stats_check_period_ms=0)


def test_tensorrt_llm_model_config_raise_error_when_invalid_request_stats_max_iterations():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`request_stats_max_iterations` must be greater than 0.",
    ):
        TensorRTLLMModelConfig(request_stats_max_iterations=0)


def test_tensorrt_llm_model_config_raise_error_when_invalid_iter_stats_max_iterations():
    with pytest.raises(
        ModelNavigatorWrongParameterError,
        match="`iter_stats_max_iterations` must be greater than 0.",
    ):
        TensorRTLLMModelConfig(iter_stats_max_iterations=0)


def test_tensorrt_llm_model_config_initialize_parameters_when_minimal_configuration_is_passed():
    config = TensorRTLLMModelConfig()

    assert len(config.parameters) == 5

    assert config.parameters["gpt_model_type"] == "inflight_batching"
    assert config.parameters["gpt_model_path"] is None
    assert config.parameters["batch_scheduler_policy"] == "max_utilization"
    assert config.parameters["FORCE_CPU_ONLY_INPUT_TENSORS"] == "no"
    assert config.parameters["executor_worker_path"] == "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"


def test_tensorrt_llm_model_config_initialize_parameters_when_empty_kv_cache_config_passed():
    config = TensorRTLLMModelConfig(kv_cache_config=KVCacheConfig())

    assert len(config.parameters) == 5

    assert config.parameters["gpt_model_type"] == "inflight_batching"
    assert config.parameters["gpt_model_path"] is None
    assert config.parameters["batch_scheduler_policy"] == "max_utilization"
    assert config.parameters["FORCE_CPU_ONLY_INPUT_TENSORS"] == "no"
    assert config.parameters["executor_worker_path"] == "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"


def test_tensorrt_llm_model_config_initialize_parameters_when_empty_peft_cache_config_passed():
    config = TensorRTLLMModelConfig(peft_cache_config=PeftCacheConfig())

    assert len(config.parameters) == 5

    assert config.parameters["gpt_model_type"] == "inflight_batching"
    assert config.parameters["gpt_model_path"] is None
    assert config.parameters["batch_scheduler_policy"] == "max_utilization"
    assert config.parameters["FORCE_CPU_ONLY_INPUT_TENSORS"] == "no"
    assert config.parameters["executor_worker_path"] == "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"


def test_tensorrt_llm_model_config_initialize_parameters_when_additional_configuration_is_passed():
    config = TensorRTLLMModelConfig(
        encoder_dir=pathlib.Path("/tmp/encoder_dir"),
        max_beam_width=10,
        gpu_device_ids=[0, 1, 2, 3],
        gpu_weights_percent=0.5,
        decoding_mode=DecodingMode.MEDUSA,
        medusa_choices=[[1, 2, 3]],
        enable_chunked_context=False,
        normalize_log_probs=False,
        cancellation_check_period_ms=10,
        stats_check_period_ms=20,
        exclude_input_in_output=False,
        request_stats_max_iterations=12,
        iter_stats_max_iterations=15,
        kv_cache_config=KVCacheConfig(
            enable_block_reuse=False,
            max_tokens=14,
            sink_token_length=16,
            max_attention_window=18,
            free_gpu_memory_fraction=0.1,
            host_cache_size=100,
            onboard_blocks=110,
        ),
        peft_cache_config=PeftCacheConfig(
            optimal_adapter_size=10,
            max_adapter_size=20,
            gpu_memory_fraction=0.5,
            host_memory_bytes=1024,
        ),
    )

    assert len(config.parameters) == 29

    assert config.parameters["gpt_model_type"] == "inflight_batching"
    assert config.parameters["gpt_model_path"] is None
    assert config.parameters["batch_scheduler_policy"] == "max_utilization"
    assert config.parameters["FORCE_CPU_ONLY_INPUT_TENSORS"] == "no"
    assert config.parameters["executor_worker_path"] == "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"

    assert config.parameters["encoder_model_path"] == pathlib.Path("/tmp/encoder_dir")
    assert config.parameters["max_beam_width"] == 10
    assert config.parameters["gpu_device_ids"] == "0,1,2,3"
    assert config.parameters["gpu_weights_percent"] == 0.5
    assert config.parameters["decoding_mode"] == DecodingMode.MEDUSA.value
    assert config.parameters["medusa_choices"] == "[[1, 2, 3]]"
    assert config.parameters["enable_chunked_context"] is False
    assert config.parameters["normalize_log_probs"] is False
    assert config.parameters["cancellation_check_period_ms"] == 10
    assert config.parameters["stats_check_period_ms"] == 20
    assert config.parameters["exclude_input_in_output"] is False
    assert config.parameters["request_stats_max_iterations"] == 12
    assert config.parameters["iter_stats_max_iterations"] == 15
    assert config.parameters["enable_kv_cache_reuse"] is False
    assert config.parameters["max_tokens_in_paged_kv_cache"] == 14
    assert config.parameters["sink_token_length"] == 16
    assert config.parameters["max_attention_window_size"] == 18
    assert config.parameters["kv_cache_free_gpu_mem_fraction"] == 0.1
    assert config.parameters["kv_cache_host_memory_bytes"] == 100
    assert config.parameters["kv_cache_onboard_blocks"] == 110
    assert config.parameters["lora_cache_optimal_adapter_size"] == 10
    assert config.parameters["lora_cache_max_adapter_size"] == 20
    assert config.parameters["lora_cache_gpu_memory_fraction"] == 0.5
    assert config.parameters["lora_cache_host_memory_bytes"] == 1024


def test_tensorrt_llm_model_config_update_parameters_when_engine_dir_passed():
    config = TensorRTLLMModelConfig()

    assert config.parameters["gpt_model_path"] is None

    config.engine_dir = pathlib.Path("/tmp/engine_dir")

    assert config.parameters["gpt_model_path"] == "/tmp/engine_dir"
