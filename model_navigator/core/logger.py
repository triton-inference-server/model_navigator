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
"""Logger module."""

import contextlib
import inspect
import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys
from functools import lru_cache
from multiprocessing import current_process
from typing import Dict, Optional, TextIO, Tuple, Union

from loguru import logger

from model_navigator.configuration.constants import (
    NAVIGATOR_LOG_FORMAT_ENV,
    NAVIGATOR_LOG_LEVEL_ENV,
    NAVIGATOR_LOGGER_NAME,
    NAVIGATOR_THIRD_PARTY_LOG_LEVEL_ENV,
    NAVIGATOR_USE_MULTIPROCESSING,
    OUTPUT_LOGS_FLAG,
)
from model_navigator.utils.environment import get_console_output, use_multiprocessing

LOGGER = logger.bind(**{NAVIGATOR_LOGGER_NAME: True})


class InterceptHandler(logging.Handler):
    """Handler to catch all logging module logs and push to loguru logger."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        # Get corresponding Loguru level if it exists.
        level: Union[str, int]
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where <originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


@lru_cache
def get_navigator_log_level() -> str:
    """Returns logging level."""
    return os.environ.get(NAVIGATOR_LOG_LEVEL_ENV, "INFO").upper()


@lru_cache
def get_third_party_log_level() -> str:
    """Returns logging level."""
    return os.environ.get(NAVIGATOR_THIRD_PARTY_LOG_LEVEL_ENV, "WARNING").upper()


@lru_cache
def get_log_format():
    """Returns log format."""
    formats = {
        "compact": (
            "<green>{time:HH:mm:ss}</green>|<level>{level:.1}</level>|<cyan>{module}</cyan>|<level>{message}</level>"
        ),
        "normal": (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> |  "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        "verbose": (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | {process.name} | "
            "<cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    }

    log_format = os.environ.get(NAVIGATOR_LOG_FORMAT_ENV, "verbose")
    if log_format not in formats:
        raise Exception(f"Illegal logging format specified. Use one of: {list(formats.keys())}")

    return formats[log_format]


def navigator_record_predicate(record: Dict) -> bool:
    """Returns True if log emitted by navigator logger."""
    return NAVIGATOR_LOGGER_NAME in record["extra"]


def third_party_record_predicate(record: Dict) -> bool:
    """Returns True if log emitted by 3rd party library."""
    # Import here to avoid circular imports
    from model_navigator.core.memory_logging import gpu_memory_record_predicate

    return not navigator_record_predicate(record) and not gpu_memory_record_predicate(record)


def forward_python_logging_to_loguru() -> None:
    """Use intercept handler to capture all holds and forward to loguru."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def forward_polygraphy_logging_to_python_logging() -> None:
    """Reconfigures polygraphy logger to use python logging instead of stdout/stderr."""
    # TODO: configure polygraphy logger - waiting for polygraphy next release
    # from polygraphy.logger import G_LOGGER
    # G_LOGGER.use_python_logging_system = True
    ...


def configure_logging_sink(sink: Union[TextIO, str, pathlib.Path]) -> Tuple[int, int]:
    """Configures given sink for the loguru."""
    navigator_sink_id = logger.add(
        sink,
        level=get_navigator_log_level(),
        format=get_log_format(),
        filter=navigator_record_predicate,
        enqueue=True,
    )
    third_party_sink_id = logger.add(
        sink,
        level=get_third_party_log_level(),
        format=get_log_format(),
        filter=third_party_record_predicate,
        enqueue=True,
    )
    return navigator_sink_id, third_party_sink_id


def configure_initial_logging() -> None:
    """Configures initial logging before workspace or anything else is ready.

    This should be called on once in the main process only.
    """
    logging.captureWarnings(True)
    forward_polygraphy_logging_to_python_logging()
    forward_python_logging_to_loguru()
    logger.remove()  # remove pre-configured logger
    configure_logging_sink(sys.stderr)
    logger.debug("Initial logging has been configured")


def reconfigure_logging_to_file(log_path: pathlib.Path) -> None:
    """Reconfigures logging system to log everything to the log_dir in the workspace.

    Note: this can be called in the main process and in the child process.

    Args:
        log_path: A path where to store logs
    """
    forward_polygraphy_logging_to_python_logging()
    forward_python_logging_to_loguru()

    # inform user that we are now switching logging to file only in parent process.
    process_name = current_process().name
    if process_name == "MainProcess":
        logger.info("Logs will be stored to the file: {}", log_path)

    # reconfigure logging
    logger.remove()  # remove existing configuration
    configure_logging_sink(log_path)

    # configure GPU memory logging to a separate file
    from model_navigator.core.memory_logging import configure_gpu_memory_logging_sink

    gpu_memory_log_path = log_path.parent / "gpu_memory.log"
    configure_gpu_memory_logging_sink(gpu_memory_log_path)

    if OUTPUT_LOGS_FLAG in get_console_output():
        configure_logging_sink(sys.stderr)

    logger.info("{} starts logging to {}", process_name, log_path)


if current_process().name == "MainProcess":
    # configure main logging system in main process during imports
    # spawn method must be used for windows and because of cuda initialization
    if (method := mp.get_start_method(allow_none=True)) is None:
        mp.set_start_method("spawn")
    elif method == "fork":
        if use_multiprocessing():
            raise Exception(
                "Model Navigator requires running conversions and exports in child processes using spawn mode to "
                "have better isolation for errors. However, some code has already set multiprocessing to fork mode. "
                "You can either paste the following code at the beginning of your imports to force "
                "spawn method: import multiprocessing;multiprocessing.set_start_method('spawn'). "
                "Or you can set the following environment variable to force running everything in a single process "
                f"(at the cost of no error isolation): {NAVIGATOR_USE_MULTIPROCESSING}=False"
            )

    configure_initial_logging()
else:
    # child processes must not log anything to stdout/stderr - remove all loggers during initialization
    logger.remove()


# TODO:: should be deprecated once trt logging to python logger works
class StdoutLogger:
    """Context manager to redirect stdout to logger."""

    def __init__(self, logger, level=logging.INFO):
        """Initialize the StdoutLOgger context manager."""
        self.logger = logger
        self.name = self.logger.name
        self.level = level
        self._redirect_stdout = contextlib.redirect_stdout(self)  # pytype: disable=wrong-arg-types

    def write(self, msg):
        """Write message to logger."""
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self):
        """Flush pass method."""
        pass

    def __enter__(self):
        """Enter the context manager."""
        self._redirect_stdout.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self._redirect_stdout.__exit__(exc_type, exc_value, traceback)


class LoggingContext(contextlib.AbstractContextManager):
    """LoggingContext to handle correct logging options when commands are executed.

    Example of use:
        log_dir = pathlib.Path("/path/to/log/directory")
        with LoggingContext(log_dir=log_dir, command_name="ExampleCommand"):
            LOGGER.info("Log inside the context")
    """

    def __init__(
        self,
        *,
        log_dir: Optional[pathlib.Path] = None,
        command_name: Optional[str] = None,
        runner_cls=None,
        model_config=None,
    ):
        """Initialize the context.

        Args:
            log_dir: Optional path to directory where log file is stored.
            command_name: Optional name of the command being executed.
            runner_cls: Optional runner class from execution unit.
            model_config: Optional model configuration from execution unit.
        """
        self.sink_ids = None
        self.gpu_memory_sink_id = None
        self.command_name = command_name
        self.runner_cls = runner_cls
        self.model_config = model_config
        self.initial_memory_info = None
        self.initial_host_info = None

        if log_dir:
            # Import here to avoid circular imports
            from model_navigator.core.memory_logging import configure_gpu_memory_logging_sink

            log_dir.mkdir(parents=True, exist_ok=True)
            self.sink_ids = configure_logging_sink(log_dir / "format.log")
            self.gpu_memory_sink_id = configure_gpu_memory_logging_sink(log_dir / "gpu_memory.log")

    def __enter__(self):
        """Enter the context and capture initial GPU and host memory usage without logging."""
        # Import here to avoid circular imports
        from model_navigator.core.memory_logging import get_memory_info

        # Just capture memory info without logging
        self.initial_memory_info, self.initial_host_info = get_memory_info()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: F841
        """Exit the context, log all memory usage in nested hierarchy, and clean handlers.

        Args:
            exc_type: class of exception
            exc_value: type of exception
            traceback: traceback of exception
        """
        # Import here to avoid circular imports
        from model_navigator.core.memory_logging import log_command_gpu_memory_usage

        # Log GPU and host memory usage information
        log_command_gpu_memory_usage(
            initial_memory_info=self.initial_memory_info,
            initial_host_info=self.initial_host_info,
            command_name=self.command_name,
            runner_cls=self.runner_cls,
            model_config=self.model_config,
        )

        # Remove logging sink handlers
        if self.sink_ids is not None:
            [logger.remove(sink_id) for sink_id in self.sink_ids]

        if self.gpu_memory_sink_id is not None and self.gpu_memory_sink_id != 0:
            logger.remove(self.gpu_memory_sink_id)


def log_dict(title: str, data: Dict):
    """Log dictionary data with provided tittle.

    Args:
        title: The title for logged information
        data: The dictionary with content to log

    """
    LOGGER.info(pad_string(title))
    LOGGER.info(json.dumps(data, indent=4))


def pad_string(s: str, width: int = 60) -> str:
    """Pad string with `=` signs.

    Args:
        s (str): String.
        width (int): Width of the string.

    Returns:
        str: Padded string.
    """
    s = f" {s} "
    return s.center(width, "=")
