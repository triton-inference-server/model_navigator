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
"""Logger module."""

import contextlib
import json
import logging
import pathlib
from typing import Dict, Optional

import coloredlogs

from model_navigator.core.constants import NAVIGATOR_LOG_NAME, NAVIGATOR_LOGGER_NAME

LOGGER = logging.getLogger(NAVIGATOR_LOGGER_NAME)
LOGGER.propagate = False
log_format = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

coloredlogs.install(
    logger=LOGGER,
    level="INFO",
    fmt=log_format,
    field_styles={
        "asctime": {"color": "green"},
        "hostname": {"color": "magenta"},
        "levelname": {"bold": True, "color": "blue"},
        "name": {"color": "blue"},
        "programname": {"color": "cyan"},
        "username": {"color": "yellow"},
    },
    reconfigure=True,
)


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
        with LoggingContext(log_dir=log_dir):
            LOGGER.info("Log inside the context")
    """

    def __init__(
        self,
        *,
        log_dir: Optional[pathlib.Path] = None,
    ):
        """Initialize the context.

        Args:
            log_dir: Optional path to directory where log file is stored.
        """
        super().__init__()
        self.log_dir = log_dir
        self.loggers = list(logging.root.manager.loggerDict.keys())

        if self.log_dir:
            log_format = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / "format.log"
            self.log_file_handler = logging.FileHandler(log_file)
            self.log_file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(log_format)
            self.log_file_handler.setFormatter(formatter)
            LOGGER.addHandler(self.log_file_handler)

            for logger in self.loggers:
                if isinstance(logger, str):
                    logger = logging.getLogger(logger)
                if self.log_file_handler:
                    logger.addHandler(self.log_file_handler)

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: F841
        """Exit the context and clean handlers.

        Args:
            exc_type: class of exception
            exc_value: type of exception
            traceback: traceback of exception
        """
        if self.log_dir:
            for logger in self.loggers:
                if isinstance(logger, str):
                    logger = logging.getLogger(logger)
                logger.handlers = [h for h in logger.handlers if h != self.log_file_handler]
            LOGGER.removeHandler(self.log_file_handler)
            self.log_file_handler = None


def add_log_file_handler(log_dir: pathlib.Path) -> None:
    """Add log file handler to file in defined path.

    Args:
        log_dir: A path to log directory
    """
    log_file = log_dir / NAVIGATOR_LOG_NAME
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)


def get_logger_names() -> list:
    """Collect logger names as list.

    Returns:
        List with names of the loggers
    """
    return list(logging.root.manager.loggerDict.keys())


def log_dict(title: str, data: Dict):
    """Log dictionary data with provided tittle.

    Args:
        title: The title for logged information
        data: The dictionary with content to log

    """
    LOGGER.info(pad_string(title))
    LOGGER.info(json.dumps(data, indent=4))


def pad_string(s: str) -> str:
    """Pad string with `=` signs.

    Args:
        s (str): String.

    Returns:
        str: Padded string.
    """
    s = f" {s} "
    return s.center(118, "=")
