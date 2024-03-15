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
"""Common configurations for public and internal API."""

import dataclasses
import enum
import pathlib
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np

from model_navigator.exceptions import ModelNavigatorWrongParameterError

from .internal import cast_dtype, expect_type, is_shape_correct


@dataclasses.dataclass
class BaseTensorSpec:
    """Common specification of input and output tensor.

    Args:
        name: Name of the model input/output
        shape: Shape of the model input/output
        dtype: Data type
        reshape: The shape expected for this input by the backend
        is_shape_tensor: Flag marking the input is a shape tensor to the model
    """

    name: str
    shape: Tuple[int, ...]
    dtype: Optional[Union[np.dtype, Type[np.dtype]]] = None
    reshape: Tuple[int, ...] = dataclasses.field(default_factory=lambda: ())

    is_shape_tensor: Optional[bool] = False

    def __post_init__(self) -> None:
        """Validate the configuration for early error handling."""
        if self.dtype:
            self.dtype = cast_dtype(dtype=self.dtype)

        expect_type("name", self.name, str)
        expect_type("shape", self.shape, tuple)
        expect_type("reshape", self.shape, tuple, optional=True)
        expect_type("dtype", self.dtype, np.dtype, optional=True)
        expect_type("is_shape_tensor", self.is_shape_tensor, bool, optional=True)
        is_shape_correct("shape", self.shape)
        is_shape_correct("reshape", self.reshape, optional=True)


class InputTensorFormat(enum.Enum):
    """Format for input tensor.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L324

    Args:
        FORMAT_NONE: 0
        FORMAT_NHWC: 1
        FORMAT_NCHW: 2
    """

    FORMAT_NONE = 0
    FORMAT_NHWC = 1
    FORMAT_NCHW = 2


@dataclasses.dataclass
class InputTensorSpec(BaseTensorSpec):
    """Stores specification of single input tensor.

    This includes name, shape, dtype and more parameters available for input tensor in Triton Inference Server:

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L317

    Args:
        optional: Flag marking the input is optional for the model execution
        format: The format of the input.
        allow_ragged_batch:  Flag marking the input is allowed to be "ragged" in a dynamically created batch.
    """

    optional: bool = False
    format: Optional[InputTensorFormat] = None
    allow_ragged_batch: bool = False


@dataclasses.dataclass
class OutputTensorSpec(BaseTensorSpec):
    """Stores specification of single output tensor.

    This includes name, shape, dtype and more parameters available for output tensor in Triton Inference Server:

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L419

    Args:
        label_filename: The label file associated with this output.
    """

    label_filename: Optional[str] = None


class Platform(enum.Enum):
    """Define supported platforms for which model store can be created. Use to override default backends.

    Args:
        ONNXRuntimeONNX: "onnxruntime_onnx"
        TensorRTPlan: "tensorrt_plan"
        PyTorchLibtorch: "pytorch_libtorch"
        TensorFlowGraphDef: "tensorflow_graphdef"
        TensorFlowSavedModel: "tensorflow_savedmodel"
    """

    ONNXRuntimeONNX = "onnxruntime_onnx"
    TensorRTPlan = "tensorrt_plan"
    PyTorchLibtorch = "pytorch_libtorch"
    TensorFlowGraphDef = "tensorflow_graphdef"
    TensorFlowSavedModel = "tensorflow_savedmodel"


class DeviceKind(enum.Enum):
    """Device kind for model deployment.

    Read more in Triton Inference server [model configuration]
    [model configuration]:https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L150

    Args:
        KIND_AUTO: "KIND_AUTO"
        KIND_CPU: "KIND_CPU"
        KIND_GPU: "KIND_GPU"
    """

    KIND_AUTO = "KIND_AUTO"
    KIND_CPU = "KIND_CPU"
    KIND_GPU = "KIND_GPU"


class TimeoutAction(enum.Enum):
    """Timeout action definition for timeout_action QueuePolicy field.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1044

    Args:
        REJECT: "REJECT"
        DELAY: "DELAY"
    """

    REJECT = "REJECT"
    DELAY = "DELAY"


@dataclasses.dataclass
class InstanceGroup:
    """Configuration for model instance group.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L143

    Args:
        kind: Kind of this instance group.
        count: For a group assigned to GPU, the number of instances created for
               each GPU listed in 'gpus'. For a group assigned to CPU the number
               of instances created.
        name: Optional name of this group of instances.
        gpus: GPU(s) where instances should be available.
        passive: Whether the instances within this instance group will be accepting
                 inference requests from the scheduler.
        host_policy: The host policy name that the instance to be associated with.
        profile: For TensorRT models containing multiple optimization profile, this
                 parameter specifies a set of optimization profiles available to this
                 instance group.
    """

    kind: Optional[DeviceKind] = None
    count: Optional[int] = None
    name: Optional[str] = None
    gpus: List[int] = dataclasses.field(default_factory=lambda: [])
    passive: bool = False
    host_policy: Optional[str] = None
    profile: List[str] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        """Validate the configuration for early error handling."""
        if self.count is not None and self.count < 1:
            raise ModelNavigatorWrongParameterError("The `count` must be greater or equal 1.")

        if self.kind not in [None, DeviceKind.KIND_GPU, DeviceKind.KIND_AUTO] and len(self.gpus) > 0:
            raise ModelNavigatorWrongParameterError(
                f"`gpus` cannot be set when device is not {DeviceKind.KIND_GPU} or {DeviceKind.KIND_AUTO}"
            )


@dataclasses.dataclass
class QueuePolicy:
    """Model queue policy configuration.

    Used for `default_queue_policy` and `priority_queue_policy` fields in DynamicBatcher configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1037

    Args:
        timeout_action: The action applied to timed-out request.
        default_timeout_microseconds: The default timeout for every request, in microseconds.
        allow_timeout_override: Whether individual request can override the default timeout value.
        max_queue_size: The maximum queue size for holding requests.
    """

    timeout_action: TimeoutAction = TimeoutAction.REJECT
    default_timeout_microseconds: int = 0
    allow_timeout_override: bool = False
    max_queue_size: int = 0


class TensorRTOptPrecision(enum.Enum):
    """TensorRT optimization allowed precision.

    Args:
        FP16: fp16 precision
        FP32: fp32 precision
    """

    FP16 = "fp16"
    FP32 = "fp32"


@dataclasses.dataclass
class TensorRTAccelerator:
    """TensorRT accelerator configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L871

    Args:
        precision: The precision used for optimization
        max_workspace_size: The maximum GPU memory the model can use temporarily during execution
        max_cached_engines: The maximum number of cached TensorRT engines in dynamic TensorRT ops
        minimum_segment_size: The smallest model subgraph that will be considered for optimization by TensorRT
    """

    precision: TensorRTOptPrecision = TensorRTOptPrecision.FP32
    max_workspace_size: Optional[int] = None
    max_cached_engines: Optional[int] = None
    minimum_segment_size: Optional[int] = None


@dataclasses.dataclass
class DynamicBatcher:
    """Dynamic batching configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1104

    Args:
        max_queue_delay_microseconds: The maximum time, in microseconds, a request will be delayed in
                                      the scheduling queue to wait for additional requests for batching.
        preferred_batch_size: Preferred batch sizes for dynamic batching.
        preserve_ordering : Should the dynamic batcher preserve the ordering of responses to
                            match the order of requests received by the scheduler.
        priority_levels: The number of priority levels to be enabled for the model.
        default_priority_level: The priority level used for requests that don't specify their priority.
        default_queue_policy: The default queue policy used for requests.
        priority_queue_policy: Specify the queue policy for the priority level.
    """

    max_queue_delay_microseconds: int = 0
    preferred_batch_size: Optional[list] = None
    preserve_ordering: bool = False
    priority_levels: int = 0
    default_priority_level: int = 0
    default_queue_policy: Optional[QueuePolicy] = None
    priority_queue_policy: Optional[Dict[int, QueuePolicy]] = None

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.default_priority_level > self.priority_levels:
            raise ModelNavigatorWrongParameterError(
                "The `default_priority_level` must be between 1 and " f"{self.priority_levels}."
            )

        if self.priority_queue_policy:
            if not self.priority_levels:
                raise ModelNavigatorWrongParameterError(
                    "Provide the `priority_levels` if you want to define `priority_queue_policy` "
                    "for Dynamic Batching."
                )

            for priority in self.priority_queue_policy.keys():
                if priority < 0 or priority > self.priority_levels:
                    raise ModelNavigatorWrongParameterError(
                        f"Invalid `priority`={priority} provided. The value must be between "
                        f"1 and {self.priority_levels}."
                    )


class SequenceBatcherControlKind(enum.Enum):
    """Sequence Batching control options.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1193

    Args:
        CONTROL_SEQUENCE_START: "CONTROL_SEQUENCE_START"
        CONTROL_SEQUENCE_READY: "CONTROL_SEQUENCE_READY"
        CONTROL_SEQUENCE_END: "CONTROL_SEQUENCE_END"
        CONTROL_SEQUENCE_CORRID: "CONTROL_SEQUENCE_CORRID"
    """

    CONTROL_SEQUENCE_START = "CONTROL_SEQUENCE_START"
    CONTROL_SEQUENCE_READY = "CONTROL_SEQUENCE_READY"
    CONTROL_SEQUENCE_END = "CONTROL_SEQUENCE_END"
    CONTROL_SEQUENCE_CORRID = "CONTROL_SEQUENCE_CORRID"


@dataclasses.dataclass
class SequenceBatcherControl:
    """Sequence Batching control configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1186

    Args:
        kind: The kind of this control.
        dtype: The control's datatype.
        int32_false_true: The control's true and false setting is indicated by setting
                          a value in an int32 tensor.
        fp32_false_true: The control's true and false setting is indicated by setting
                         a value in a fp32 tensor.
        bool_false_true: The control's true and false setting is indicated by setting
                         a value in a bool tensor.
    """

    kind: SequenceBatcherControlKind
    dtype: Optional[Union[np.dtype, Type[np.dtype]]] = None
    int32_false_true: List[int] = dataclasses.field(default_factory=lambda: [])
    fp32_false_true: List[float] = dataclasses.field(default_factory=lambda: [])
    bool_false_true: List[bool] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.kind == SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID and self.dtype is None:
            raise ModelNavigatorWrongParameterError(f"The {self.kind} control type requires `dtype` to be specified.")

        if self.kind == SequenceBatcherControlKind.CONTROL_SEQUENCE_CORRID and any([
            self.int32_false_true,
            self.fp32_false_true,
            self.bool_false_true,
        ]):
            raise ModelNavigatorWrongParameterError(
                f"The {self.kind} control type requires `dtype` to be specified only."
            )

        controls = [
            SequenceBatcherControlKind.CONTROL_SEQUENCE_START,
            SequenceBatcherControlKind.CONTROL_SEQUENCE_END,
            SequenceBatcherControlKind.CONTROL_SEQUENCE_READY,
        ]

        if self.kind in controls and self.dtype:
            raise ModelNavigatorWrongParameterError(f"The {self.kind} control does not support `dtype` parameter.")

        if self.kind in controls and not (self.int32_false_true or self.fp32_false_true or self.bool_false_true):
            raise ModelNavigatorWrongParameterError(
                f"The {self.kind} control type requires one of: "
                "`int32_false_true`, `fp32_false_true`, `bool_false_true` to be specified."
            )

        if self.int32_false_true and len(self.int32_false_true) != 2:
            raise ModelNavigatorWrongParameterError(
                "The `int32_false_true` field should be two element list with false and true values. Example: [0 , 1]"
            )

        if self.fp32_false_true and len(self.fp32_false_true) != 2:
            raise ModelNavigatorWrongParameterError(
                "The `fp32_false_true` field should be two element list with false and true values. Example: [0 , 1]"
            )

        if self.bool_false_true and len(self.bool_false_true) != 2:
            raise ModelNavigatorWrongParameterError(
                "The `bool_false_true` field should be two element list with false and true values. "
                "Example: [False, True]"
            )

        if self.dtype:
            self.dtype = cast_dtype(dtype=self.dtype)

        expect_type("dtype", self.dtype, np.dtype, optional=True)


@dataclasses.dataclass
class SequenceBatcherControlInput:
    """Sequence Batching control input configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1283

    Args:
        input_name: The name of the model input.
        controls: List of  control value(s) that should be communicated to the
                  model using this model input.
    """

    input_name: str
    controls: List[SequenceBatcherControl]


@dataclasses.dataclass
class SequenceBatcherInitialState:
    """Sequence Batching initial state configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1304

    Args:
        name:
        shape: The shape of the state tensor, not including the batch dimension.
        dtype: The data-type of the state.
        zero_data: The identifier for using zeros as initial state data.
        data_file: The file whose content will be used as the initial data for
                   the state in row-major order.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: Optional[Union[np.dtype, Type[np.dtype]]] = None
    zero_data: Optional[bool] = None
    data_file: Optional[pathlib.Path] = None

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if not self.zero_data and not self.data_file:
            raise ModelNavigatorWrongParameterError("zero_data or data_file has to be defined. None was provided.")

        if self.zero_data and self.data_file:
            raise ModelNavigatorWrongParameterError("zero_data or data_file has to be defined. Both were provided.")

        if self.dtype:
            self.dtype = cast_dtype(dtype=self.dtype)

        expect_type("name", self.name, str)
        expect_type("shape", self.shape, tuple)
        expect_type("dtype", self.dtype, np.dtype, optional=True)
        is_shape_correct("shape", self.shape)


@dataclasses.dataclass
class SequenceBatcherState:
    """Sequence Batching state configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1353

    Args:
        input_name: The name of the model state input.
        output_name: The name of the model state output.
        dtype: The data-type of the state.
        shape: The shape of the state tensor.
        initial_states: The optional field to specify the list of initial states for the model.
    """

    input_name: str
    output_name: str
    dtype: Union[np.dtype, Type[np.dtype]]
    shape: Tuple[int, ...]
    initial_states: List[SequenceBatcherInitialState] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        self.dtype = cast_dtype(dtype=self.dtype)

        expect_type("shape", self.shape, tuple)
        expect_type("dtype", self.dtype, np.dtype, optional=True)
        is_shape_correct("shape", self.shape)


@dataclasses.dataclass
class SequenceBatcherStrategyDirect:
    """Sequence Batching strategy direct configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1394

    Args:
        max_queue_delay_microseconds: The maximum time, in microseconds, a candidate request
                                      will be delayed in the sequence batch scheduling queue to
                                      wait for additional requests for batching.
        minimum_slot_utilization: The minimum slot utilization that must be satisfied to
                                  execute the batch before 'max_queue_delay_microseconds' expires.
    """

    max_queue_delay_microseconds: int = 0
    minimum_slot_utilization: float = 0.0


@dataclasses.dataclass
class SequenceBatcherStrategyOldest:
    """Sequence Batching strategy oldest configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1431

    Args:
        max_candidate_sequences:  Maximum number of candidate sequences that the batcher maintains.
        preferred_batch_size: Preferred batch sizes for dynamic batching of candidate sequences.
        max_queue_delay_microseconds: The maximum time, in microseconds, a candidate request
                                      will be delayed in the dynamic batch scheduling queue to
                                      wait for additional requests for batching.
    """

    max_candidate_sequences: int
    preferred_batch_size: List[int] = dataclasses.field(default_factory=lambda: [])
    max_queue_delay_microseconds: int = 0


@dataclasses.dataclass
class SequenceBatcher:
    """Sequence batching configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1179

    Args:
        strategy: The strategy used by the sequence batcher.
        max_sequence_idle_microseconds: The maximum time, in microseconds, that a sequence is allowed to
                                        be idle before it is aborted.
        control_inputs: The model input(s) that the server should use to communicate
                        sequence start, stop, ready and similar control values to the model.
        states: The optional state that can be stored in Triton for performing
                inference requests on a sequence.
    """

    strategy: Optional[Union[SequenceBatcherStrategyDirect, SequenceBatcherStrategyOldest]] = None
    max_sequence_idle_microseconds: Optional[int] = None
    control_inputs: List[SequenceBatcherControlInput] = dataclasses.field(default_factory=lambda: [])
    states: List[SequenceBatcherState] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.strategy and (
            not isinstance(self.strategy, SequenceBatcherStrategyDirect)
            and not isinstance(self.strategy, SequenceBatcherStrategyOldest)
        ):
            raise ModelNavigatorWrongParameterError("Unsupported strategy type provided.")


class ModelWarmupInputDataType(enum.Enum):
    """Model warmup input data type.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1605

    Args:
        ZERO: "ZERO"
        RANDOM: "RANDOM"
        FILE: "FILE"
    """

    ZERO = "ZERO"
    RANDOM = "RANDOM"
    FILE = "FILE"


@dataclasses.dataclass
class ModelWarmupInput:
    """Model warmup input configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1605

    Args:
        shape: Shape of the model input/output
        dtype: Data type
        input_data_type: Type of input data used for warmup
        input_data_file: Path to file with input data. Provide the path where the file is located.
                         Required only when input_data_type is `ModelWarmupInputDataType.DATA_FILE`
    """

    shape: Tuple[int, ...]
    dtype: Optional[Union[np.dtype, Type[np.dtype]]]
    input_data_type: ModelWarmupInputDataType
    input_data_file: Optional[pathlib.Path] = None

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.input_data_type == ModelWarmupInputDataType.FILE and self.input_data_file is None:
            raise ModelNavigatorWrongParameterError("`input_data_file` is required. Set the file path.")

        if self.input_data_type != ModelWarmupInputDataType.FILE and self.input_data_file is not None:
            raise ModelNavigatorWrongParameterError("`input_data_file` is not required. Remove the parameter.")


@dataclasses.dataclass
class ModelWarmup:
    """Model warmup configuration.

    Read more in Triton Inference server [model configuration]
    [model configuration]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1598

    Args:
        batch_size: The batch size of the inference request. This must be >= 1. For models that don't support batching,
                    batch_size must be 1.
        inputs: The warmup meta data associated with every model input, including control tensors.
        iterations: The number of iterations that this warmup sample will be executed. For example, if this field is
                    set to 2, 2 model executions using this sample will be scheduled for warmup. Default value is 0 which
                    indicates that this sample will be used only once.
    """

    inputs: Dict[str, ModelWarmupInput]
    batch_size: int = 1
    iterations: int = 0

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.batch_size < 1:
            raise ModelNavigatorWrongParameterError("`batch_size` must be greater or equal 1.")

        if self.iterations < 0:
            raise ModelNavigatorWrongParameterError("`iterations` must be greater or equal 0.")
