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
from typing import List, Optional

import numba

from model_navigator.device.gpu_device_factory import GPUDeviceFactory


def get_gpus(gpus: Optional[List[str]] = None):
    """
    Creates a list of GPU UUIDs corresponding to the GPUs visible to
    model_analyzer.
    """
    gpus = gpus or ["all"]

    deployer_gpus = []
    if len(gpus) == 1 and gpus[0] == "all":
        devices = numba.cuda.list_devices()
        for device in devices:
            gpu_device = GPUDeviceFactory.create_device_by_cuda_index(device.id)
            deployer_gpus.append(str(gpu_device.device_uuid(), encoding="ascii"))
    else:
        devices = gpus
        for device in devices:
            gpu_device = GPUDeviceFactory.create_device_by_uuid(device)
            deployer_gpus.append(str(gpu_device.device_uuid(), encoding="ascii"))

    return deployer_gpus
