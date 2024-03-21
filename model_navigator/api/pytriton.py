# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Public API definition for PyTriton related functionality."""

import dataclasses
import enum
from typing import Dict, List, Optional, Type, Union

import numpy as np

from model_navigator.api.config import TensorType
from model_navigator.exceptions import ModelNavigatorNotFoundError
from model_navigator.package.package import Package
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runtime_analyzer.strategy import MaxThroughputAndMinLatencyStrategy, RuntimeSearchStrategy


class TimeoutAction(enum.Enum):
    """Timeout action definition for timeout_action QueuePolicy field.

    Args:
        REJECT (str): Reject the request and return error message accordingly.
        DELAY (str): Delay the request until all other requests at the same (or higher) priority levels
            that have not reached their timeouts are processed.
    """

    REJECT: str = "REJECT"
    DELAY: str = "DELAY"


@dataclasses.dataclass
class QueuePolicy:
    """Model queue policy configuration.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1037

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


@dataclasses.dataclass
class DynamicBatcher:
    """Dynamic batcher configuration.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1104

    Args:
        max_queue_delay_microseconds: The maximum time, in microseconds, a request will be delayed in
                                      the scheduling queue to wait for additional requests for batching.
        preferred_batch_size: Preferred batch sizes for dynamic batching.
        preserve_ordering: Should the dynamic batcher preserve the ordering of responses to
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


@dataclasses.dataclass
class ModelConfig:
    """Additional model configuration for running model through Triton Inference Server.

    Args:
        batching: Flag to enable/disable batching for model.
        max_batch_size: The maximal batch size that would be handled by model.
        batcher: Configuration of Dynamic Batching for the model.
        response_cache: Flag to enable/disable response cache for the model
        decoupled: Flag to enable/disable decoupled transaction policy
    """

    batching: bool = True
    max_batch_size: int = 4
    batcher: DynamicBatcher = dataclasses.field(default_factory=DynamicBatcher)
    response_cache: bool = False
    decoupled: bool = False


@dataclasses.dataclass(frozen=True)
class Tensor:
    """Model input and output definition for Triton deployment.

    Args:
        shape: Shape of the input/output tensor.
        dtype: Data type of the input/output tensor.
        name: Name of the input/output of model.
        optional: Flag to mark if input is optional.
    """

    shape: tuple
    dtype: Union[np.dtype, Type[np.dtype], Type[object]]
    name: Optional[str] = None
    optional: Optional[bool] = False


class PyTritonAdapter:
    """Provides model and configuration for PyTrtion deployment."""

    def __init__(
        self,
        package: Package,
        strategy: Optional[RuntimeSearchStrategy] = None,
        runner_return_type: TensorType = TensorType.NUMPY,
    ):
        """Initialize PyTritonAdapter.

        Args:
            package: A package object to be searched for best possible model.
            strategy: Strategy for finding the best model. Defaults to `MaxThroughputAndMinLatencyStrategy`
            runner_return_type: The type of the output tensor. Defaults to `TensorType.NUMPY`.
                If the return_type supports CUDA tensors (e.g. TensorType.TORCH) and the input tensors are on CUDA,
                there will be no additional data transfer between CPU and GPU.
        """
        self._package = package
        self._strategy = MaxThroughputAndMinLatencyStrategy() if strategy is None else strategy
        self._runner = self._package.get_runner(strategy=self._strategy, return_type=runner_return_type)
        self._batching = self._package.status.config.get("batch_dim", None) == 0

    @property
    def batching(self) -> bool:
        """Returns status of batching support by the runner.

        Returns:
            True if runner supports batching, False otherwise.
        """
        return self._batching

    @property
    def runner(self) -> NavigatorRunner:
        """Returns runner.

            Runner must be activated before use with activate() method.

        Returns:
            Model Navigator runner.
        """
        return self._runner

    @property
    def inputs(self) -> List[Tensor]:
        """Returns inputs configuration.

        Returns:
            List with Tensor objects describing inputs configuration of runner
        """
        inputs = []
        for input in self._runner._input_metadata.values():
            inputs.append(
                Tensor(
                    shape=input.shape if not self._batching else input.shape[1:],
                    dtype=input.dtype.type,
                    name=input.name,
                    optional=input.optional,
                )
            )
        return inputs

    @property
    def outputs(self) -> List[Tensor]:
        """Returns outputs configuration.

        Returns:
            List with Tensor objects describing outputs configuration of runner
        """
        outputs = []
        for output in self._runner._output_metadata.values():
            outputs.append(
                Tensor(
                    shape=output.shape if not self._batching else output.shape[1:],
                    dtype=output.dtype.type,
                    name=output.name,
                    optional=output.optional,
                )
            )
        return outputs

    @property
    def config(self) -> ModelConfig:
        """Returns config for pytriton.

        Returns:
            ModelConfig with configuration for PyTrtion bind method.

        """
        model_status = self._package.get_best_model_status(strategy=self._strategy)
        if model_status:
            bs_from_profiling = max(
                r.batch_size
                for r in model_status.runners_status[self._runner.name()].result["Performance"]["profiling_results"]
            )
        else:
            raise ModelNavigatorNotFoundError(f"Cannot find model status for strategy: {self._strategy}")

        return ModelConfig(max_batch_size=bs_from_profiling)
