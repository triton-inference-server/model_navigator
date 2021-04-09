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
import numba

from .gpu_device_factory import GPUDeviceFactory
from ..config import ModelNavigatorBaseConfig


def get_gpus(config: ModelNavigatorBaseConfig):
    """
    Creates a list of GPU UUIDs corresponding to the GPUs visible to
    model_analyzer.
    """

    deployer_gpus = []
    if len(config.gpus) == 1 and config.gpus[0] == "all":
        devices = numba.cuda.list_devices()
        for device in devices:
            gpu_device = GPUDeviceFactory.create_device_by_cuda_index(device.id)
            deployer_gpus.append(str(gpu_device.device_uuid(), encoding="ascii"))
    else:
        devices = config.gpus
        for device in devices:
            gpu_device = GPUDeviceFactory.create_device_by_uuid(device)
            deployer_gpus.append(str(gpu_device.device_uuid(), encoding="ascii"))

    return deployer_gpus
