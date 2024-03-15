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
"""Device utils."""

import ctypes
import logging
import uuid
from ctypes import c_uint8
from typing import List, Optional, Sequence, Union

from model_navigator.exceptions import ModelNavigatorError

LOGGER = logging.getLogger(__name__)
UUID_SIZE = 16
MAX_NAME_SIZE = 256
CUDA_SUCCESS = 0

try:
    cuda = ctypes.cdll.LoadLibrary("libcuda.so")
except OSError as e:
    LOGGER.warning(f"CUDA not available: {e}")
    cuda = None


def _check_ret(err):
    if err != CUDA_SUCCESS:
        err_str = ctypes.c_char_p()
        cuda.cuGetErrorString(err, ctypes.byref(err_str))
        raise ModelNavigatorError(f"CUDA error: {err_str.value}.")


def _get_device_count():
    count = ctypes.c_int()
    _check_ret(cuda.cuDeviceGetCount(ctypes.byref(count)))
    return count.value


def get_available_gpus():
    """Get all available GPUs in the system."""
    if cuda is None:
        return []

    _check_ret(cuda.cuInit(0))
    device_count = _get_device_count()
    dev_uuid = bytearray(UUID_SIZE)
    name_buf = bytearray(MAX_NAME_SIZE)
    devices = []
    for device in range(device_count):
        _check_ret(cuda.cuDeviceGetUuid((c_uint8 * UUID_SIZE).from_buffer(dev_uuid), device))
        _check_ret(cuda.cuDeviceGetName((c_uint8 * MAX_NAME_SIZE).from_buffer(name_buf), MAX_NAME_SIZE, device))
        entry = {
            "name": name_buf.decode().rstrip("\x00"),
            "uuid": f"GPU-{str(uuid.UUID(bytes=bytes(dev_uuid)))}",
        }
        devices.append(entry)
    return devices


def get_gpus(gpus: Optional[List[Union[int, str]]]):
    """Creates a list of GPU UUIDs corresponding to the GPUs visible to Model Navigator.

    GPU UUIDs are returned as hex-encoded strings prefixed with "GPU-", e.g.:
      "GPU-11111111-1111-1111-1111-111111111111"
    """
    if not gpus:
        return []

    devices = [dev["uuid"] for dev in get_available_gpus()]

    requested_gpus = gpus or ("all",)
    if len(requested_gpus) == 1 and requested_gpus[0] == "all":
        requested_gpus = range(len(devices))

    navigator_gpus = []
    for gpu in requested_gpus:
        try:
            device = int(gpu)
            navigator_gpus.append(devices[device])
        except (ValueError, IndexError):
            if gpu not in devices:
                raise ModelNavigatorError(f"GPU {gpu} was not found.") from None
            navigator_gpus.append(gpu)
    return navigator_gpus


def _get_cuda_driver_version():
    if cuda is None:
        return "N/A"
    version = ctypes.c_int()
    _check_ret(cuda.cuDriverGetVersion(ctypes.byref(version)))
    return f"{version.value // 1000}.{version.value % 100}"


def _get_nvidia_driver_version():
    try:
        with open("/proc/driver/nvidia/version") as f:
            return f.readline().split(":")[1].strip()
    except Exception:
        return "N/A"


def get_environment_info(gpus: Optional[Sequence[str]] = None):
    """Get information bout the gpus and drivers."""
    info = {
        "driver": _get_nvidia_driver_version(),
        "cuda_driver": _get_cuda_driver_version(),
        "gpus": get_available_gpus(),
        "gpus_used": get_gpus(gpus),
    }
    # TODO: "container_tag"
    return info


def is_cuda_available():
    """Return True if CUDA available, False otherwise."""
    return bool(get_gpus(["all"]))
