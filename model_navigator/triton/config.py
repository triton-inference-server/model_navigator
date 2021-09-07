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
# See the License for the specific languag
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from model_navigator.utils.config import BaseConfig


class ModelControlMode(Enum):
    EXPLICIT = "explicit"
    POLL = "poll"


class DeviceKind(Enum):
    CPU = "cpu"
    GPU = "gpu"


class TensorRTOptPrecision(Enum):
    FP16 = "fp16"
    FP32 = "fp32"


class BackendAccelerator(Enum):
    NONE = "none"  # backward compatibility for CLI
    AMP = "amp"
    TRT = "trt"


@dataclass
class TritonClientConfig(BaseConfig):
    server_url: str = "grpc://localhost:8001"


@dataclass
class TritonModelOptimizationConfig(BaseConfig):
    backend_accelerator: Optional[BackendAccelerator] = dataclasses.field(default=None)
    tensorrt_precision: Optional[TensorRTOptPrecision] = dataclasses.field(default=None)
    tensorrt_capture_cuda_graph: bool = False


@dataclass
class TritonModelSchedulerConfig(BaseConfig):
    max_batch_size: int = 32
    preferred_batch_sizes: Optional[List[int]] = None
    max_queue_delay_us: int = 0


@dataclass
class TritonModelInstancesConfig(BaseConfig):
    engine_count_per_device: Dict[DeviceKind, int] = field(default_factory=lambda: {})
