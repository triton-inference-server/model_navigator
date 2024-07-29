# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Configuration of TensorRT platform supported models on Triton Inference Server."""

import dataclasses
import enum
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from model_navigator.exceptions import ModelNavigatorWrongParameterError

from .base_model_config import BaseSpecializedModelConfig
from .common import DeviceKind, DynamicBatcher, InputTensorSpec, InstanceGroup, OutputTensorSpec
from .internal import Backend


class BatchingStrategy(enum.Enum):
    """Define the supported batch strategies."""

    INFLIGHT = "inflight_batching"
    STATIC = "v1"


class BatchSchedulerPolicy(enum.Enum):
    """Define the supported batch scheduler policies."""

    MAX_UTILIZATION = "max_utilization"
    GUARANTEED_NO_EVICT = "guaranteed_no_evict"


class DecodingMode(enum.Enum):
    """Define the supported decoding modes."""

    AUTO = "auto"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TOP_K_TOP_P = "top_k_top_p"
    BEAM_SEARCH = "beam_search"
    MEDUSA = "medusa"


@dataclasses.dataclass()
class KVCacheConfig:
    """Configuration of KV cache in TRT-LLM.

    More: https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html#_CPPv4N12tensorrt_llm8executor13KvCacheConfigE

    Args:
        enable_block_reuse: Controls if KV cache blocks can be reused for different requests.
        max_tokens: The maximum number of tokens that should be stored in the KV cache If both max_tokens and
                    free_gpu_memory_fraction are specified, memory corresponding to the minimum will be allocated.
        sink_token_length: Number of sink tokens (tokens to always keep in attention window)
        max_attention_window: Size of the attention window for each sequence. Only the last max_attention_window tokens
                              of each sequence will be stored in the KV cache.
        free_gpu_memory_fraction: The fraction of GPU memory fraction that should be allocated for the KV cache.
                                  Default is 90%. If both max_tokens and free_gpu_memory_fraction are specified,
                                  memory corresponding to the minimum will be allocated.
        host_cache_size: Size of secondary memory pool in bytes. Default is 0. Having a secondary memory pool increases
                         KV cache block reuse potential.
        onboard_blocks: Controls whether offloaded blocks should be onboarded back into primary memory before
                        being reused.
    """

    enable_block_reuse: Optional[bool] = None  # enable_kv_cache_reuse
    max_tokens: Optional[int] = None  # max_tokens_in_paged_kv_cache
    sink_token_length: Optional[int] = None  # sink_token_length
    max_attention_window: Optional[int] = None  # max_attention_window_size
    free_gpu_memory_fraction: Optional[float] = None  # kv_cache_free_gpu_mem_fraction
    host_cache_size: Optional[int] = None  # kv_cache_host_memory_bytes
    onboard_blocks: Optional[int] = None  # kv_cache_onboard_blocks

    _MAPPING = {
        "enable_block_reuse": "enable_kv_cache_reuse",
        "max_tokens": "max_tokens_in_paged_kv_cache",
        "max_attention_window": "max_attention_window_size",
        "free_gpu_memory_fraction": "kv_cache_free_gpu_mem_fraction",
        "host_cache_size": "kv_cache_host_memory_bytes",
        "onboard_blocks": "kv_cache_onboard_blocks",
    }

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ModelNavigatorWrongParameterError("`max_tokens` must be greater than 0.")

        if self.sink_token_length is not None and self.sink_token_length <= 0:
            raise ModelNavigatorWrongParameterError("`sink_token_length` must be greater than 0.")

        if self.max_attention_window is not None and self.max_attention_window <= 0:
            raise ModelNavigatorWrongParameterError("`max_attention_window` must be greater than 0.")

        if self.free_gpu_memory_fraction is not None and (
            self.free_gpu_memory_fraction < 0.0 or self.free_gpu_memory_fraction > 1.0
        ):
            raise ModelNavigatorWrongParameterError("`free_gpu_memory_fraction` must be between 0.0 and 1.0.")

        if self.host_cache_size is not None and self.host_cache_size <= 0:
            raise ModelNavigatorWrongParameterError("`host_cache_size` must be greater than 0.")

        if self.onboard_blocks is not None and self.onboard_blocks <= 0:
            raise ModelNavigatorWrongParameterError("`onboard_blocks` must be greater than 0.")

    def as_parameters(self):
        """Convert dataclass to configuration flags passed to backend as parameters."""
        data = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue

            mapped_key = self._MAPPING.get(k, k)
            data[mapped_key] = v

        return data


@dataclasses.dataclass()
class PeftCacheConfig:
    """Configuration of Peft Cache.

    More: https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html#_CPPv4N12tensorrt_llm8executor15PeftCacheConfigE

    Args:
        optimal_adapter_size:
        max_adapter_size:
        gpu_memory_fraction:
        host_memory_bytes:
    """

    optimal_adapter_size: Optional[int] = None
    max_adapter_size: Optional[int] = None
    gpu_memory_fraction: Optional[float] = None
    host_memory_bytes: Optional[int] = None

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        if self.optimal_adapter_size is not None and self.optimal_adapter_size <= 0:
            raise ModelNavigatorWrongParameterError("`optimal_adapter_size` must be greater than 0.")

        if self.max_adapter_size is not None and self.max_adapter_size <= 0:
            raise ModelNavigatorWrongParameterError("`max_adapter_size` must be greater than 0.")

        if self.max_adapter_size and self.optimal_adapter_size and self.max_adapter_size < self.optimal_adapter_size:
            raise ModelNavigatorWrongParameterError(
                "`max_adapter_size` must be greater than or equal to `optimal_adapter_size`."
            )

        if self.gpu_memory_fraction is not None and (self.gpu_memory_fraction < 0.0 or self.gpu_memory_fraction > 1.0):
            raise ModelNavigatorWrongParameterError("`gpu_memory_fraction` must be between 0.0 and 1.0.")

        if self.host_memory_bytes is not None and self.host_memory_bytes <= 0:
            raise ModelNavigatorWrongParameterError("`host_memory_bytes` must be greater than 0.")

    def as_parameters(self):
        """Convert dataclass to configuration flags passed to backend as parameters."""
        data = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue

            data[f"lora_cache_{k}"] = v

        return data


@dataclasses.dataclass()
class TensorRTLLMModelConfig(BaseSpecializedModelConfig):
    """Specialized model config for TensorRT-LLM platform supported model.

    Adapted from TensorRT-LLM config: https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxts

    Relevant TensorRT-LLM classes:
    - ExecutorConfig: https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html#_CPPv4N12tensorrt_llm8executor14ExecutorConfigE
    - KVCacheConfig: https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html#_CPPv4N12tensorrt_llm8executor13KvCacheConfigE
    - PeftCacheConfig: https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html#_CPPv4N12tensorrt_llm8executor15PeftCacheConfigE

    Args:
        engine_dir: Path to the TensorRT engine directory.
        encoder_dir: Path to the encoder model directory.
        max_beam_width: Maximal size of each beam in beam search.
        batching_strategy: Batching strategy for model.
        batch_scheduler_policy: Batching scheduler policy for model.
        decoding_mode: Decoding mode for model.
        gpu_device_ids: List of GPU devices on which model is running.
        gpu_weights_percent: The percentage of GPU memory fraction that should be allocated for weights.
        kv_cache_config: KV cache config for model.
        peft_cache_config: Peft cache config for model.
        enable_chunked_context: Enable chunked context for model
        normalize_log_probs: Controls if log probabilities should be normalized or not.
        cancellation_check_period_ms: The request cancellation period check in ms.
        stats_check_period_ms: The statistics checking period in ms.
        request_stats_max_iterations: Controls the maximum number of iterations for which to keep per-request statistics.
        iter_stats_max_iterations: Controls the maximum number of iterations for which to keep statistics.
        exclude_input_in_output: Controls if output tokens in Result should include the input tokens. Default is false.
        medusa_choices: Medusa choices as in https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py
    """

    encoder_dir: Optional[pathlib.Path] = None
    max_beam_width: Optional[int] = None
    batching_strategy: BatchingStrategy = BatchingStrategy.INFLIGHT
    batch_scheduler_policy: BatchSchedulerPolicy = BatchSchedulerPolicy.MAX_UTILIZATION
    decoding_mode: Optional[DecodingMode] = None
    gpu_device_ids: List[int] = dataclasses.field(default_factory=lambda: [])
    gpu_weights_percent: Optional[float] = None
    kv_cache_config: Optional[KVCacheConfig] = None
    peft_cache_config: Optional[PeftCacheConfig] = None
    enable_chunked_context: Optional[bool] = None
    normalize_log_probs: Optional[bool] = None
    cancellation_check_period_ms: Optional[int] = None
    stats_check_period_ms: Optional[int] = None
    request_stats_max_iterations: Optional[int] = None
    iter_stats_max_iterations: Optional[int] = None
    exclude_input_in_output: Optional[bool] = None
    medusa_choices: Optional[Union[List[int], List[List[int]], List[Tuple[int]]]] = None

    _engine_dir: Optional[pathlib.Path] = None

    _CUSTOM_FIELDS = [
        "_engine_dir",
        "engine_dir",
        "encoder_dir",
        "max_beam_width",
        "batching_strategy",
        "batch_scheduler_policy",
        "decoding_mode",
        "gpu_device_ids",
        "gpu_weights_percent",
        "kv_cache_config",
        "peft_cache_config",
        "enable_chunked_context",
        "normalize_log_probs",
        "cancellation_check_period_ms",
        "cancellation_check_period_ms",
        "stats_check_period_ms",
        "request_stats_max_iterations",
        "iter_stats_max_iterations",
        "exclude_input_in_output",
        "medusa_choices",
    ]

    def __post_init__(self):
        """Validate the configuration for early error handling."""
        super().__post_init__()
        self._validate_config()
        self._initialize_instance_groups()
        self._initialize_inputs()
        self._initialize_outputs()
        self._initialize_parameters()

    @property
    def backend(self) -> Backend:
        """Define backend value for config."""
        return Backend.TensorRTLLM

    @property
    def engine_dir(self) -> Optional[pathlib.Path]:
        """Engined directory path."""
        return self._engine_dir

    @engine_dir.setter
    def engine_dir(self, engine_dir: pathlib.Path) -> None:
        """Engined directory path."""
        self._engine_dir = engine_dir
        self.parameters["gpt_model_path"] = self._engine_dir.as_posix()

    def _validate_config(self):
        if self.instance_groups:
            raise ModelNavigatorWrongParameterError(
                "Instance groups are not supported for TensorRT-LLM backend has predefined instance groups."
            )

        if self.inputs:
            raise ModelNavigatorWrongParameterError(
                "Inputs parameters is not supported as TensorRT-LLM backend has predefined shapes."
            )

        if self.outputs:
            raise ModelNavigatorWrongParameterError(
                "Outputs parameters is not supported as TensorRT-LLM backend has predefined shapes."
            )

        if self.default_model_filename is not None:
            raise ModelNavigatorWrongParameterError("Default model filename is not supported.")

        if not isinstance(self.batcher, DynamicBatcher):
            raise ModelNavigatorWrongParameterError("Batcher must be an instance of DynamicBatcher.")

        if self.decoding_mode == DecodingMode.MEDUSA and not self.medusa_choices:
            raise ModelNavigatorWrongParameterError("Medusa choices are required for Medusa decoding mode.")

        if self.medusa_choices:
            if not isinstance(self.medusa_choices, list) and not isinstance(self.medusa_choices, tuple):
                raise ModelNavigatorWrongParameterError(
                    """Medusa choices must be a list or tuple. See an example at: """
                    """https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py."""
                )

            for item in self.medusa_choices:
                if not isinstance(item, list) and not isinstance(item, tuple):
                    raise ModelNavigatorWrongParameterError(
                        """Medusa choices item must be a list or tuple. See an example at: """
                        """https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py."""
                    )

        if self.gpu_weights_percent is not None and (self.gpu_weights_percent < 0.0 or self.gpu_weights_percent > 1.0):
            raise ModelNavigatorWrongParameterError("`gpu_weights_percent` must be between 0.0 and 1.0.")

        if self.cancellation_check_period_ms is not None and self.cancellation_check_period_ms <= 0:
            raise ModelNavigatorWrongParameterError("`cancellation_check_period_ms` must be greater than 0.")

        if self.stats_check_period_ms is not None and self.stats_check_period_ms <= 0:
            raise ModelNavigatorWrongParameterError("`stats_check_period_ms` must be greater than 0.")

        if self.request_stats_max_iterations is not None and self.request_stats_max_iterations <= 0:
            raise ModelNavigatorWrongParameterError("`request_stats_max_iterations` must be greater than 0.")

        if self.iter_stats_max_iterations is not None and self.iter_stats_max_iterations <= 0:
            raise ModelNavigatorWrongParameterError("`iter_stats_max_iterations` must be greater than 0.")

    def _initialize_instance_groups(self):
        self.instance_groups = [InstanceGroup(kind=DeviceKind.KIND_CPU, count=1)]

    def _initialize_inputs(self):
        self.inputs = [
            InputTensorSpec(
                name="input_ids",
                dtype=np.int32,
                shape=(-1,),
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="input_lengths",
                dtype=np.int32,
                shape=(1,),
                reshape=(),
            ),
            InputTensorSpec(
                name="request_output_len",
                dtype=np.int32,
                shape=(1,),
            ),
            InputTensorSpec(
                name="draft_input_ids",
                dtype=np.int32,
                shape=(-1,),
                optional=True,
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="decoder_input_ids",
                dtype=np.int32,
                shape=(-1,),
                optional=True,
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="decoder_input_lengths",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="draft_logits",
                dtype=np.int32,
                shape=(-1, -1),
                optional=True,
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="draft_acceptance_threshold",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="end_id",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="pad_id",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="bad_words_list",
                dtype=np.int32,
                shape=(2, -1),
                optional=True,
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="embedding_bias",
                dtype=np.float32,
                shape=(-1,),
                optional=True,
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="beam_width",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="temperature",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="runtime_top_k",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="runtime_top_p",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="runtime_top_p_min",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="runtime_top_p_decay",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="runtime_top_p_reset_ids",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="len_penalty",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="early_stopping",
                dtype=np.bool_,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="repetition_penalty",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="min_length",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="beam_search_diversity_rate",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="presence_penalty",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="frequency_penalty",
                dtype=np.float32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="random_seed",
                dtype=np.uint64,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="return_log_probs",
                dtype=np.bool_,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="return_context_logits",
                dtype=np.bool_,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="return_generation_logits",
                dtype=np.bool_,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="stop",
                dtype=np.bool_,
                shape=(1,),
                optional=True,
            ),
            InputTensorSpec(
                name="streaming",
                dtype=np.bool_,
                shape=(1,),
                optional=True,
            ),
            InputTensorSpec(
                name="prompt_embedding_table",
                dtype=np.float16,
                shape=(-1, -1),
                optional=True,
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="prompt_vocab_size",
                dtype=np.int32,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="lora_task_id",
                dtype=np.uint64,
                shape=(1,),
                optional=True,
                reshape=(),
            ),
            InputTensorSpec(
                name="lora_weights",
                dtype=np.float16,
                shape=(-1, -1),
                optional=True,
                allow_ragged_batch=True,
            ),
            InputTensorSpec(
                name="lora_config",
                dtype=np.int32,
                shape=(-1, 3),
                optional=True,
                allow_ragged_batch=True,
            ),
        ]

    def _initialize_outputs(self):
        self.outputs = [
            OutputTensorSpec(name="output_ids", dtype=np.int32, shape=(-1, -1)),
            OutputTensorSpec(name="sequence_length", dtype=np.int32, shape=(-1,)),
            OutputTensorSpec(name="cum_log_probs", dtype=np.float32, shape=(-1,)),
            OutputTensorSpec(name="output_log_probs", dtype=np.float32, shape=(-1, -1)),
            OutputTensorSpec(name="context_logits", dtype=np.float32, shape=(-1, -1)),
            OutputTensorSpec(name="generation_logits", dtype=np.float32, shape=(-1, -1, -1)),
        ]

    def _initialize_parameters(self):
        self.parameters: Dict[str, Any] = {
            "gpt_model_type": self.batching_strategy.value,
            "gpt_model_path": self.engine_dir,
            "batch_scheduler_policy": self.batch_scheduler_policy.value,
            "FORCE_CPU_ONLY_INPUT_TENSORS": "no",
            "executor_worker_path": "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker",
        }

        if self.encoder_dir:
            self.parameters["encoder_model_path"] = self.encoder_dir

        if self.max_beam_width:
            self.parameters["max_beam_width"] = self.max_beam_width

        if self.gpu_device_ids:
            self.parameters["gpu_device_ids"] = ",".join([str(device_id) for device_id in self.gpu_device_ids])

        if self.gpu_weights_percent:
            self.parameters["gpu_weights_percent"] = self.gpu_weights_percent

        if self.decoding_mode:
            self.parameters["decoding_mode"] = self.decoding_mode.value

        if self.enable_chunked_context is not None:
            self.parameters["enable_chunked_context"] = self.enable_chunked_context

        if self.normalize_log_probs is not None:
            self.parameters["normalize_log_probs"] = self.normalize_log_probs

        if self.cancellation_check_period_ms is not None:
            self.parameters["cancellation_check_period_ms"] = self.cancellation_check_period_ms

        if self.stats_check_period_ms is not None:
            self.parameters["stats_check_period_ms"] = self.stats_check_period_ms

        if self.exclude_input_in_output is not None:
            self.parameters["exclude_input_in_output"] = self.exclude_input_in_output

        if self.request_stats_max_iterations is not None:
            self.parameters["request_stats_max_iterations"] = self.request_stats_max_iterations

        if self.iter_stats_max_iterations is not None:
            self.parameters["iter_stats_max_iterations"] = self.iter_stats_max_iterations

        if self.medusa_choices is not None:
            self.parameters["medusa_choices"] = json.dumps(self.medusa_choices)

        if self.peft_cache_config is not None:
            lora_params = self.peft_cache_config.as_parameters()
            self.parameters = {**self.parameters, **lora_params}

        if self.kv_cache_config is not None:
            kv_cache_config_params = self.kv_cache_config.as_parameters()
            self.parameters = {**self.parameters, **kv_cache_config_params}
