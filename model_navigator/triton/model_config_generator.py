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
"""Generator class for creating Triton model config.

The class consume the ModelConfig object as a constructor argument and produce the Triton model config in form of
dict or file.

    Typical usage example:

        model_config = ModelConfig(model_name="simple")
        generator = ModelConfigGenerator(model_config)
        generator.to_file("/path/to/config.pbtxt")
"""

import json
import logging
import pathlib
from typing import Dict, List, Union

import numpy as np
from google.protobuf import json_format, text_format  # pytype: disable=pyi-error
from tritonclient import utils as client_utils  # noqa: F401
from tritonclient.grpc import model_config_pb2  # pytype: disable=import-error

from model_navigator.utils.common import dataclass2dict

from .model_config import ModelConfig
from .specialized_configs import (
    AutoMixedPrecisionAccelerator,
    BaseTensorSpec,
    DynamicBatcher,
    GPUIOAccelerator,
    InputTensorSpec,
    ModelWarmupInput,
    ModelWarmupInputDataType,
    ONNXOptimization,
    OpenVINOAccelerator,
    OutputTensorSpec,
    SequenceBatcher,
    SequenceBatcherControlInput,
    SequenceBatcherState,
    SequenceBatcherStrategyDirect,
    SequenceBatcherStrategyOldest,
    TensorFlowOptimization,
    TensorRTAccelerator,
    TensorRTOptimization,
)

LOGGER = logging.getLogger(__name__)


class ModelConfigGenerator:
    """Generate the protobuf config from ModelConfig object."""

    def __init__(self, config: ModelConfig):
        """Initialize generator.

        Args:
            config: model config object
        """
        self._config = config

    def to_file(self, config_path: Union[str, pathlib.Path]) -> str:
        """Serialize ModelConfig to prototxt and save to config_path directory.

        Args:
            config_path: path to configuration file

        Returns:
            A string with generated model configuration
        """
        # https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
        model_config = self._get_config()
        LOGGER.debug(f"Generated Triton config:\n{json.dumps(model_config, indent=4)}")

        config_payload = json_format.ParseDict(model_config, model_config_pb2.ModelConfig())
        LOGGER.debug(f"Generated Triton config payload:\n{config_payload}")

        config_path = pathlib.Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        model_config_bytes = text_format.MessageToBytes(config_payload)

        # WAR: triton requires max_batch_size = 0 to be explicit written
        # while this is not stored in payload during MessageToBytes
        if model_config["max_batch_size"] == 0:
            model_config_bytes += b"max_batch_size: 0\n"

        with config_path.open("wb") as cfg:
            cfg.write(model_config_bytes)

        LOGGER.debug(f"Generated config stored in {config_path}")

        return config_payload

    def _get_config(self) -> Dict:
        """Create a Triton model config from ModelConfig object.

        Returns:
            Dict with model configuration data
        """
        model_config = {
            "name": self._config.model_name,
        }
        if self._config.platform is not None:
            model_config["platform"] = self._config.platform.value
            LOGGER.info(
                f"Platform provided. Using platform {self._config.platform.value} even if backend was provided."
            )
        elif self._config.backend is not None:
            model_config["backend"] = self._config.backend.value

        if self._config.default_model_filename is not None:
            model_config["default_model_filename"] = self._config.default_model_filename

        self._set_batching(model_config)
        self._set_model_signature(model_config)
        self._set_instance_group(model_config)
        self._set_optimization(model_config)
        self._set_parameters(model_config)
        self._set_response_cache(model_config)
        self._set_decoupled_policy(model_config)
        self._set_model_warmup(model_config)

        return model_config

    def _set_batching(self, model_config: Dict) -> None:
        """Configure batching for model deployment on Triton Inference Server.

        Args:
            model_config: Dict with model config for Triton Inference Server
        """
        if not self._config.batching:
            model_config["max_batch_size"] = 0
            LOGGER.debug("Batching for model is disabled. The `max_batch_size` field value set to 0.")
            return

        model_config["max_batch_size"] = self._config.max_batch_size
        if isinstance(self._config.batcher, DynamicBatcher):
            self._set_dynamic_batching(
                batcher=self._config.batcher,
                model_config=model_config,
            )
        elif isinstance(self._config.batcher, SequenceBatcher):
            self._set_sequence_batching(
                batcher=self._config.batcher,
                model_config=model_config,
            )

    def _set_dynamic_batching(self, batcher: DynamicBatcher, model_config: Dict) -> None:
        """Configure dynamic batching for model deployment on Triton Inference Server.

        Args:
            batcher: Dynamic batcher object with configuration
            model_config: Dict with model config for Triton Inference Server
        """
        dynamic_batching_config = {}
        if batcher.max_queue_delay_microseconds > 0:
            dynamic_batching_config["maxQueueDelayMicroseconds"] = int(batcher.max_queue_delay_microseconds)

        if batcher.preferred_batch_size:
            dynamic_batching_config["preferredBatchSize"] = [int(bs) for bs in batcher.preferred_batch_size]

        if batcher.preserve_ordering:
            dynamic_batching_config["preserveOrdering"] = batcher.preserve_ordering

        if batcher.priority_levels:
            dynamic_batching_config["priorityLevels"] = batcher.priority_levels

        if batcher.default_priority_level:
            dynamic_batching_config["defaultPriorityLevel"] = batcher.default_priority_level

        if batcher.default_queue_policy:
            priority_queue_policy_config = {
                "timeoutAction": batcher.default_queue_policy.timeout_action.value,
                "defaultTimeoutMicroseconds": int(batcher.default_queue_policy.default_timeout_microseconds),
                "allowTimeoutOverride": batcher.default_queue_policy.allow_timeout_override,
                "maxQueueSize": int(batcher.default_queue_policy.max_queue_size),
            }
            dynamic_batching_config["defaultQueuePolicy"] = priority_queue_policy_config

        if batcher.priority_queue_policy:
            priority_queue_policy_config = {}
            for priority, queue_policy in batcher.priority_queue_policy.items():
                priority_queue_policy_config[priority] = {
                    "timeoutAction": queue_policy.timeout_action.value,
                    "defaultTimeoutMicroseconds": int(queue_policy.default_timeout_microseconds),
                    "allowTimeoutOverride": queue_policy.allow_timeout_override,
                    "maxQueueSize": int(queue_policy.max_queue_size),
                }

            dynamic_batching_config["priorityQueuePolicy"] = priority_queue_policy_config

        model_config["dynamic_batching"] = dynamic_batching_config

    def _set_sequence_batching(self, batcher: SequenceBatcher, model_config: Dict) -> None:
        """Configure sequence batching for model deployment on Triton Inference Server.

        Args:
            batcher: Sequence batcher object with configuration
            model_config: Dict with model config for Triton Inference Server
        """
        sequence_batching_config = {}

        if batcher.strategy:
            self._set_sequence_batcher_strategy(
                strategy=batcher.strategy,
                sequence_batching_config=sequence_batching_config,
            )

        if batcher.control_inputs:
            self._set_sequence_batcher_control_input(
                control_inputs=batcher.control_inputs,
                sequence_batching_config=sequence_batching_config,
            )
        if batcher.states:
            self._set_sequence_batcher_state(
                states=batcher.states,
                sequence_batching_config=sequence_batching_config,
            )
        model_config["sequence_batching"] = sequence_batching_config

    def _set_sequence_batcher_strategy(
        self,
        strategy: Union[SequenceBatcherStrategyDirect, SequenceBatcherStrategyOldest],
        sequence_batching_config: Dict,
    ):
        """Configure sequence batcher strategy for model.

        Args:
            strategy: Strategy to configure
            sequence_batching_config: Dictionary where configuration is stored
        """
        if isinstance(strategy, SequenceBatcherStrategyOldest):
            strategy_oldest_data = {"maxCandidateSequences": strategy.max_candidate_sequences}
            if strategy.preferred_batch_size:
                strategy_oldest_data["preferredBatchSize"] = strategy.preferred_batch_size

            if strategy.max_queue_delay_microseconds:
                strategy_oldest_data["maxQueueDelayMicroseconds"] = strategy.max_queue_delay_microseconds

            sequence_batching_config["oldest"] = strategy_oldest_data
        elif isinstance(strategy, SequenceBatcherStrategyDirect):
            strategy_direct_data = {}
            if strategy.max_queue_delay_microseconds:
                strategy_direct_data["maxQueueDelayMicroseconds"] = strategy.max_queue_delay_microseconds

            if strategy.minimum_slot_utilization:
                strategy_direct_data["minimumSlotUtilization"] = strategy.minimum_slot_utilization

            sequence_batching_config["direct"] = strategy_direct_data

    def _set_sequence_batcher_control_input(
        self,
        control_inputs: List[SequenceBatcherControlInput],
        sequence_batching_config: Dict,
    ):
        """Configure sequence batcher control input for model.

        Args:
            control_inputs: List of control input to configure
            sequence_batching_config: Dictionary where configuration is stored
        """
        control_inputs_data = []
        for control_input in control_inputs:
            controls_data = []
            for control in control_input.controls:
                control_data = {
                    "kind": control.kind.value,
                }
                if control.dtype:
                    control_data["data_type"] = self._format_data_type(control.dtype)
                if control.fp32_false_true:
                    control_data["fp32_false_true"] = control.fp32_false_true
                if control.int32_false_true:
                    control_data["int32_false_true"] = control.int32_false_true
                if control.bool_false_true:
                    control_data["bool_false_true"] = control.bool_false_true

                controls_data.append(control_data)

            control_input_data = {
                "name": control_input.input_name,
                "control": controls_data,
            }

            control_inputs_data.append(control_input_data)

        sequence_batching_config["control_input"] = control_inputs_data

    def _set_sequence_batcher_state(self, states: List[SequenceBatcherState], sequence_batching_config: Dict):
        """Configure sequence batcher state for model.

        Args:
            states: List of states to configure
            sequence_batching_config: Dictionary where configuration is stored
        """
        states_data = []
        for state in states:
            initial_states_data = []
            for initial_state in state.initial_states:
                initial_state_data = {
                    "name": initial_state.name,
                    "dims": list(initial_state.shape),
                    "data_type": self._format_data_type(initial_state.dtype),
                }
                if initial_state.zero_data:
                    initial_state_data["zero_data"] = True
                elif initial_state.data_file:
                    initial_state_data["data_file"] = initial_state.data_file.name

                initial_states_data.append(initial_state_data)

            state_data = {
                "input_name": state.input_name,
                "output_name": state.output_name,
                "dims": list(state.shape),
                "data_type": self._format_data_type(state.dtype),
            }
            if initial_states_data:
                state_data["initial_state"] = initial_states_data

            states_data.append(state_data)

        sequence_batching_config["state"] = states_data

    def _set_instance_group(self, model_config: Dict) -> None:
        """Configure instance group for model deployment on Triton Inference Server.

        Args:
            model_config: Dict with model config for Triton Inference Server
        """
        instance_groups = []
        for instance_group in self._config.instance_groups:
            instance_group_dict = dataclass2dict(instance_group)
            instance_group_dict = self._filter_empty_values(instance_group_dict)
            instance_groups.append(instance_group_dict)

        if instance_groups:
            model_config["instance_group"] = instance_groups

    def _set_parameters(self, model_config: Dict) -> None:
        """Configure backend parameters for model deployment on Triton Inference Server.

        Args:
            model_config: Dict with model config for Triton Inference Server
        """
        parameters = {}
        for key, value in self._config.parameters.items():
            parameters[key] = {
                "string_value": str(value),
            }

        if parameters:
            model_config["parameters"] = parameters

    def _set_model_signature(self, model_config: Dict) -> None:
        """Configure model signature  for model deployment on Triton Inference Server.

        Args:
            model_config: Dict with model config for Triton Inference Server

        """
        if self._config.inputs:
            model_config["input"] = [self._rewrite_input_tensor_spec(spec) for spec in self._config.inputs]

        if self._config.outputs:
            model_config["output"] = [self._rewrite_output_tensor_spec(spec) for spec in self._config.outputs]

    def _set_optimization(self, model_config: Dict):
        """Configure backend accelerators.

        Args:
            model_config: Model config to append optimizations to
        """
        if not self._config.optimization:
            return

        if isinstance(self._config.optimization, TensorRTOptimization):
            self._set_tensorrt_optimization(
                optimization=self._config.optimization,
                model_config=model_config,
            )
        elif isinstance(self._config.optimization, ONNXOptimization):
            self._set_onnx_optimization(
                optimization=self._config.optimization,
                model_config=model_config,
            )
        elif isinstance(self._config.optimization, TensorFlowOptimization):
            self._set_tensorflow_optimization(
                optimization=self._config.optimization,
                model_config=model_config,
            )

    def _set_tensorrt_optimization(self, optimization: TensorRTOptimization, model_config: Dict):
        """Configure TensorRT optimization.

        Args:
            optimization: Optimization configuration
            model_config: Dictionary where configuration has to be stored
        """
        optimization_config = {}
        if optimization.cuda_graphs:
            optimization_config["cuda"] = {"graphs": optimization.cuda_graphs}

        if optimization:
            optimization_config["eager_batching"] = True

        threshold = optimization.gather_kernel_buffer_threshold
        if threshold:
            optimization_config["gather_kernel_buffer_threshold"] = threshold

        model_config["optimization"] = optimization_config

    def _set_onnx_optimization(self, optimization: ONNXOptimization, model_config: Dict):
        """Configure ONNX optimization.

        Args:
            optimization: Optimization configuration
            model_config: Dictionary where configuration has to be stored
        """
        optimization_config = {}
        if isinstance(optimization.accelerator, TensorRTAccelerator):
            self._set_tensorrt_accelerator(
                accelerator=optimization.accelerator,
                optimization_config=optimization_config,
            )
        elif isinstance(optimization.accelerator, OpenVINOAccelerator):
            cpu_execution_accelerator = {
                "name": "openvino",
            }
            optimization_config["execution_accelerators"] = {
                "cpu_execution_accelerator": [cpu_execution_accelerator],
            }

        model_config["optimization"] = optimization_config

    def _set_tensorflow_optimization(self, optimization: TensorFlowOptimization, model_config: Dict):
        """Configure TensorFlow optimization.

        Args:
            optimization: Optimization configuration
            model_config: Dictionary where configuration has to be stored
        """
        optimization_config = {}
        if isinstance(optimization.accelerator, TensorRTAccelerator):
            self._set_tensorrt_accelerator(
                accelerator=optimization.accelerator, optimization_config=optimization_config
            )
        elif isinstance(optimization.accelerator, AutoMixedPrecisionAccelerator):
            gpu_execution_accelerator = {
                "name": "auto_mixed_precision",
            }
            optimization_config["execution_accelerators"] = {
                "gpu_execution_accelerator": [gpu_execution_accelerator],
            }
        elif isinstance(optimization.accelerator, GPUIOAccelerator):
            gpu_execution_accelerator = {
                "name": "gpu_io",
            }
            optimization_config["execution_accelerators"] = {
                "gpu_execution_accelerator": [gpu_execution_accelerator],
            }
        model_config["optimization"] = optimization_config

    def _set_tensorrt_accelerator(self, accelerator: TensorRTAccelerator, optimization_config: Dict):
        """Configure accelerator for optimization config.

        Args:
            accelerator: Accelerator object with configuration
            optimization_config: Optimization config to append accelerators to
        """
        parameters = {}

        max_workspace_size = accelerator.max_workspace_size
        if max_workspace_size:
            max_workspace_size = max_workspace_size * 2**20
            parameters["max_workspace_size_bytes"] = str(max_workspace_size)

        precision = accelerator.precision
        if precision:
            parameters["precision_mode"] = precision.value.upper()

        max_cached_engines = accelerator.max_cached_engines
        if max_cached_engines:
            parameters["max_cached_engines"] = str(max_cached_engines)

        minimum_segment_size = accelerator.minimum_segment_size
        if minimum_segment_size:
            parameters["minimum_segment_size"] = str(minimum_segment_size)

        gpu_execution_accelerator = {"name": "tensorrt", "parameters": parameters}

        optimization_config["execution_accelerators"] = {
            "gpu_execution_accelerator": [gpu_execution_accelerator],
        }

    def _rewrite_base_tensor(self, base_tensor_spec: BaseTensorSpec) -> Dict:
        """Rewrite the base tensor configuration to dict.

        Args:
            base_tensor_spec: base tensor data to rewrite

        Returns:
            Dictionary with tensor data
        """
        if base_tensor_spec.dtype in [np.object_, np.bytes_]:
            dtype = "TYPE_STRING"
        else:
            # pytype: enable=attribute-error
            dtype = base_tensor_spec.dtype
            # pytype: enable=attribute-error
            dtype = self._format_data_type(dtype)

        tensor_spec = {
            "name": base_tensor_spec.name,
            "dims": list(base_tensor_spec.shape),
            "data_type": dtype,
        }
        if base_tensor_spec.reshape:
            tensor_spec["reshape"] = {"shape": list(base_tensor_spec.reshape)}

        if base_tensor_spec.is_shape_tensor:
            tensor_spec["is_shape_tensor"] = True

        return tensor_spec

    def _rewrite_input_tensor_spec(self, input_tensor_spec: InputTensorSpec) -> Dict:
        """Rewrite input tensor specification to dictionary.

        Args:
            input_tensor_spec: input tensor spec to rewrite

        Returns:
            Dictionary with tensor data
        """
        tensor_spec = self._rewrite_base_tensor(input_tensor_spec)

        if input_tensor_spec.optional:
            tensor_spec["optional"] = True

        if input_tensor_spec.allow_ragged_batch:
            tensor_spec["allow_ragged_batch"] = True

        if input_tensor_spec.format is not None:
            tensor_spec["format"] = input_tensor_spec.format.value

        return tensor_spec

    def _rewrite_output_tensor_spec(self, output_tensor_spec: OutputTensorSpec) -> Dict:
        """Rewrite output tensor specification to dictionary.

        Args:
            output_tensor_spec: output tensor spec to rewrite

        Returns:
            Dictionary with tensor data
        """
        tensor_spec = self._rewrite_base_tensor(output_tensor_spec)

        if output_tensor_spec.label_filename:
            tensor_spec["label_filename"] = output_tensor_spec.label_filename

        return tensor_spec

    def _set_response_cache(self, model_config: Dict):
        """Configure response cache for model.

        Args:
            model_config: Dictionary where configuration is attached.
        """
        if self._config.response_cache:
            model_config["response_cache"] = {
                "enable": self._config.response_cache,
            }

    def _set_decoupled_policy(self, model_config: Dict):
        """Configure decoupled transaction policy for model.

        Args:
            model_config: Dictionary where configuration is attached.
        """
        if self._config.decoupled:
            model_config["model_transaction_policy"] = {
                "decoupled": self._config.decoupled,
            }

    def _set_model_warmup(self, model_config: Dict):
        """Configure model warmup.

        Args:
            model_config: Dictionary where configuration is attached.
        """
        if self._config.warmup:
            warmups = []
            for name, warmup in self._config.warmup.items():
                warmups.append({
                    "name": name,
                    "batch_size": warmup.batch_size,
                    "count": warmup.iterations,
                    "inputs": {name: self._set_warmup_input(data) for name, data in warmup.inputs.items()},
                })

            model_config["model_warmup"] = warmups

    def _set_warmup_input(self, inpt: ModelWarmupInput) -> Dict:
        """Set warmup input configuration.

        Args:
            inpt: Warmup input configuration

        Returns:
            Dictionary with configuration
        """
        if inpt.dtype in [np.object_, np.bytes_]:
            dtype = "TYPE_STRING"
        else:
            # pytype: enable=attribute-error
            dtype = inpt.dtype
            # pytype: enable=attribute-error
            dtype = self._format_data_type(dtype)

        data = {
            "dims": list(inpt.shape),
            "data_type": dtype,
        }

        if inpt.input_data_type.value == ModelWarmupInputDataType.RANDOM.value:
            data["random_data"] = True
        elif inpt.input_data_type.value == ModelWarmupInputDataType.ZERO.value:
            data["zero_data"] = True
        elif inpt.input_data_type.value == ModelWarmupInputDataType.FILE.value:
            data["input_data_file"] = inpt.input_data_file.name

        return data

    def _filter_empty_values(self, data: Dict) -> Dict:
        """Filter empty dict values and return new dict.

        Args:
            data: dictionary with values to filter

        Returns:
            New dictionary without empty values
        """
        return {k: v for k, v in data.items() if v}

    def _format_data_type(self, dtype: np.dtype) -> str:
        """Format data type for model config.

        Args:
            dtype: numpy dtype object

        Returns:
            String with data type
        """
        return f"TYPE_{client_utils.np_to_triton_dtype(dtype)}"
