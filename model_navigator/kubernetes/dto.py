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
import typing


class TesterParameters(typing.NamedTuple):
    model_name: str
    batch_size: str
    triton_instances: int
    triton_gpu_engine_count: int
    triton_server_url: str


class DeployerParameters(typing.NamedTuple):
    model_name: str
    model_version: int
    format: str
    max_batch_size: int
    precision: str
    triton_gpu_engine_count: int
    triton_preferred_batch_sizes: typing.List[int]
    triton_max_queue_delay: int
    capture_cuda_graph: int
    accelerator: str
