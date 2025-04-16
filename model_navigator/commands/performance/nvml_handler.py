# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
"""NVML handler."""

from typing import ContextManager, Dict, Optional, Union

import numpy as np
from pynvml import (
    NVML_CLOCK_GRAPHICS,
    NVMLError,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
    nvmlShutdown,
)

from model_navigator.core.logger import LOGGER


class NvmlHandler(ContextManager):
    """Context manager for initializing and shutting down NVML."""

    def __init__(self) -> None:
        """Creates NVML context manager."""
        self._nvml_exists = False

    def __enter__(self) -> "NvmlHandler":
        """Initializes NVML."""
        try:
            nvmlInit()
            self._nvml_exists = True
        except NVMLError as e:
            LOGGER.debug("Unable to initialize NVML: {}", str(e))
            self._nvml_exists = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Shuts down NVML."""
        if self._nvml_exists:
            try:
                nvmlShutdown()
            except NVMLError as e:
                LOGGER.debug("Unable to shutdown NVML: {}", str(e))
            finally:
                self._nvml_exists = False

    @property
    def gpu_clock(self) -> Optional[float]:
        """Returns average gpu clock frequency in MHz for running gpus (if they exist)."""
        if not self._nvml_exists:
            return None

        gpus_running = 0
        gpu_clocks_sum = 0
        for i in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                processes = nvmlDeviceGetComputeRunningProcesses(handle)

                if len(processes) > 0:
                    gpus_running += 1
                    gpu_clocks_sum += nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
            except NVMLError as e:
                LOGGER.debug("Unable to collect NVML data for GPU {}: {}", i, str(e))
                continue

        if gpus_running == 0:
            return None

        with np.errstate(invalid="ignore"):
            return np.divide(gpu_clocks_sum, gpus_running)

    @property
    def gpu_count(self) -> int:
        """Returns number of available gpus."""
        if not self._nvml_exists:
            return 0

        try:
            return nvmlDeviceGetCount()
        except NVMLError as e:
            LOGGER.debug("Unable to collect NVML device count: {}", str(e))
            return 0

    def get_gpu_memory_info(self) -> Dict[int, Dict[str, Union[float, int, str]]]:
        """Get memory information for all available GPUs.

        Returns:
            Dictionary with GPU indices as keys and memory information as values
        """
        memory_info = {}

        if not self._nvml_exists:
            return memory_info

        for i in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                mem_info = nvmlDeviceGetMemoryInfo(handle)

                try:
                    gpu_name = nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode("utf-8")
                except NVMLError:
                    gpu_name = f"GPU {i}"

                # Convert bytes to megabytes for consistency with other logging
                memory_used_mb = mem_info.used / (1024 * 1024)
                memory_total_mb = mem_info.total / (1024 * 1024)
                memory_free_mb = mem_info.free / (1024 * 1024)

                gpu_info = {
                    "index": i,
                    "name": gpu_name,
                    "memory_used_mb": memory_used_mb,
                    "memory_total_mb": memory_total_mb,
                    "memory_free_mb": memory_free_mb,
                }
                memory_info[i] = gpu_info
            except NVMLError as e:
                LOGGER.debug("Unable to collect memory info for GPU {}: {}", i, str(e))
                continue

        return memory_info

    @property
    def gpu_memory(self) -> Dict[int, Dict[str, Union[float, int, str]]]:
        """Returns complete memory information for all GPUs.

        Returns:
            Dictionary with GPU indices as keys and complete GPU memory information as values:
            - index: GPU index
            - name: GPU name
            - memory_used_mb: Used memory in MB
            - memory_total_mb: Total memory in MB
            - memory_free_mb: Free memory in MB
        """
        return self.get_gpu_memory_info()
