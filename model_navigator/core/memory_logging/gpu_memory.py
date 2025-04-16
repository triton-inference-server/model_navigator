# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Memory logging module for GPU and Host resources."""

import os
import pathlib
from functools import lru_cache
from typing import Dict, Optional, TextIO, Tuple, Union

import psutil
from loguru import logger

from model_navigator.configuration.constants import NAVIGATOR_LOG_LEVEL_ENV
from model_navigator.utils.environment import is_env_var_set

# Environment variable to control whether to use a separate GPU memory logger
NAVIGATOR_USE_SEPARATE_GPU_MEMORY_LOG_FILE = is_env_var_set("NAVIGATOR_USE_SEPARATE_GPU_MEMORY_LOG_FILE", "0")

# Create a dedicated logger for GPU memory
if NAVIGATOR_USE_SEPARATE_GPU_MEMORY_LOG_FILE:
    GPU_MEMORY_LOGGER = logger.bind(gpu_memory=True)
else:
    # Avoid circular import by defining a function to get the logger
    def _get_logger():
        from model_navigator.core.logger import LOGGER

        return LOGGER.bind(gpu_memory=True)

    GPU_MEMORY_LOGGER = _get_logger()


@lru_cache
def get_navigator_log_level() -> str:
    """Returns logging level."""
    return os.environ.get(NAVIGATOR_LOG_LEVEL_ENV, "INFO").upper()


def gpu_memory_record_predicate(record: Dict) -> bool:
    """Returns True if log emitted by GPU memory logger."""
    return "gpu_memory" in record["extra"]


def configure_gpu_memory_logging_sink(sink: Union[TextIO, str, pathlib.Path]) -> int:
    """Configures given sink for the GPU memory logger."""
    # If not using separate GPU memory log file, return 0 (no sink added)
    if not NAVIGATOR_USE_SEPARATE_GPU_MEMORY_LOG_FILE:
        return 0

    gpu_memory_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | {process.name} | "
        "<level>{message}</level>"
    )
    return logger.add(
        sink,
        level=get_navigator_log_level(),
        format=gpu_memory_format,
        filter=gpu_memory_record_predicate,
        enqueue=True,
    )


def get_gpu_memory_info() -> Optional[Dict[int, Dict[str, Union[float, int, str]]]]:
    """Get GPU memory information.

    Returns:
        Dictionary with GPU indices as keys and memory information as values or None if error
    """
    try:
        # Import here to avoid circular imports
        from model_navigator.commands.performance.nvml_handler import NvmlHandler

        with NvmlHandler() as nvml:
            return nvml.get_gpu_memory_info()
    except Exception as e:
        # Log any error but don't fail the execution
        GPU_MEMORY_LOGGER.warning("Failed to get GPU memory information: {}", str(e))
        return None


def get_host_memory_info() -> Dict[str, float]:
    """Get host memory information.

    Returns:
        Dictionary with host memory information
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / (1024 * 1024),
            "used_mb": memory.used / (1024 * 1024),
            "free_mb": memory.available / (1024 * 1024),
            "percent": memory.percent,
        }
    except Exception as e:
        # Log any error but don't fail the execution
        GPU_MEMORY_LOGGER.warning("Failed to get host memory information: {}", str(e))
        return {
            "total_mb": 0.0,
            "used_mb": 0.0,
            "free_mb": 0.0,
            "percent": 0.0,
        }


def log_memory_info(
    memory_info: Optional[Dict[int, Dict[str, float]]],
    host_memory_info: Optional[Dict[str, float]] = None,
    prefix: str = "",
):
    """Log GPU and host memory information with indentation.

    Args:
        memory_info: Dictionary with GPU indices as keys and memory information as values
        host_memory_info: Dictionary with host memory information
        prefix: Prefix to add before each log line
    """
    if not memory_info:
        GPU_MEMORY_LOGGER.info("{}No GPU memory information available", prefix)
    else:
        for gpu_index, gpu in memory_info.items():
            used_mb = float(gpu["memory_used_mb"])
            total_mb = float(gpu["memory_total_mb"])
            free_mb = float(gpu["memory_free_mb"])
            usage_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0

            # Split memory information into separate logs with more indentation
            GPU_MEMORY_LOGGER.info("{}GPU {} ({})", prefix, gpu_index, gpu["name"])
            GPU_MEMORY_LOGGER.info("{}  Memory: {:.2f}MB used / {:.2f}MB total", prefix, used_mb, total_mb)
            GPU_MEMORY_LOGGER.info("{}  Usage: {:.2f}%", prefix, usage_percent)
            GPU_MEMORY_LOGGER.info("{}  Free: {:.2f}MB", prefix, free_mb)

    # Log host memory information
    if host_memory_info:
        GPU_MEMORY_LOGGER.info("{}Host:", prefix)
        GPU_MEMORY_LOGGER.info(
            "{}  Memory: {:.2f}MB used / {:.2f}MB total",
            prefix,
            host_memory_info["used_mb"],
            host_memory_info["total_mb"],
        )
        GPU_MEMORY_LOGGER.info("{}  Usage: {:.2f}%", prefix, host_memory_info["percent"])
        GPU_MEMORY_LOGGER.info("{}  Free: {:.2f}MB", prefix, host_memory_info["free_mb"])
    else:
        GPU_MEMORY_LOGGER.info("{}No host memory information available", prefix)


def check_memory_fully_released(
    initial_gpu_info: Optional[Dict[int, Dict[str, float]]],
    final_gpu_info: Optional[Dict[int, Dict[str, float]]],
    initial_host_info: Optional[Dict[str, float]] = None,
    final_host_info: Optional[Dict[str, float]] = None,
):
    """Check for potentially not fully released memory.

    Args:
        initial_gpu_info: Initial GPU memory information
        final_gpu_info: Final GPU memory information
        initial_host_info: Initial host memory information
        final_host_info: Final host memory information
    """
    # Check GPU memory
    if initial_gpu_info and final_gpu_info:
        for gpu_index, final_gpu in final_gpu_info.items():
            if gpu_index in initial_gpu_info:
                final_used = float(final_gpu["memory_used_mb"])
                initial_used = float(initial_gpu_info[gpu_index]["memory_used_mb"])
                memory_diff = final_used - initial_used
                if memory_diff > 0:
                    GPU_MEMORY_LOGGER.warning(
                        "  Memory not fully recovered on GPU {} - {:.2f}MB not released after command execution",
                        gpu_index,
                        memory_diff,
                    )

    # Check host memory
    if initial_host_info and final_host_info:
        memory_diff = final_host_info["used_mb"] - initial_host_info["used_mb"]
        if memory_diff > 10:  # Using threshold of 10MB to avoid false positives due to normal system activity
            GPU_MEMORY_LOGGER.warning(
                "  Memory not fully recovered on Host - {:.2f}MB not released after command execution",
                memory_diff,
            )


def log_command_gpu_memory_usage(
    initial_memory_info: Optional[Dict[int, Dict[str, float]]],
    initial_host_info: Optional[Dict[str, float]],
    command_name: Optional[str] = None,
    runner_cls=None,
    model_config=None,
):
    """Log GPU and host memory usage information for a command execution.

    Args:
        initial_memory_info: Initial GPU memory information captured before command execution
        initial_host_info: Initial host memory information captured before command execution
        command_name: Optional name of the command being executed
        runner_cls: Optional runner class from execution unit
        model_config: Optional model configuration from execution unit
    """
    # Capture final memory info
    final_memory_info = get_gpu_memory_info()
    final_host_info = get_host_memory_info()

    # Log everything in a nested hierarchy
    cmd_info = f"[{command_name}]" if command_name else ""
    GPU_MEMORY_LOGGER.info("Memory Usage {}", cmd_info)

    # Log command identification information
    if runner_cls:
        runner_name = runner_cls.name() if hasattr(runner_cls, "name") else str(runner_cls.__name__)
        GPU_MEMORY_LOGGER.info("  Runner: {}", runner_name)

    if model_config and hasattr(model_config, "key"):
        GPU_MEMORY_LOGGER.info("  Model key: {}", model_config.key)

    # Log before command memory info (nested)
    GPU_MEMORY_LOGGER.info("  Before Command:")
    log_memory_info(initial_memory_info, initial_host_info, "    ")

    # Log after command memory info (nested)
    GPU_MEMORY_LOGGER.info("  After Command:")
    log_memory_info(final_memory_info, final_host_info, "    ")

    # Check memory fully released
    check_memory_fully_released(initial_memory_info, final_memory_info, initial_host_info, final_host_info)

    return final_memory_info, final_host_info


def get_memory_info() -> Tuple[Optional[Dict[int, Dict[str, Union[float, int, str]]]], Dict[str, float]]:
    """Get both GPU and host memory information.

    Returns:
        Tuple containing GPU memory information dictionary and host memory information
    """
    return get_gpu_memory_info(), get_host_memory_info()
