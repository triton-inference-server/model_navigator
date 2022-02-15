# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import coloredlogs

LOGGER = logging.getLogger(__name__)


def init_logger(*, verbose: bool = False, colored_logs: bool = True):
    set_logger(verbose=bool(verbose), colored_logs=colored_logs)
    set_tf_verbosity(verbose=bool(verbose))
    LOGGER.debug(f"initialized logger verbose={verbose}")


def set_tf_verbosity(verbose: bool = False):
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    tf_cpp_min_log_level = 0 if verbose else 1
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(tf_cpp_min_log_level)
    os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"


def set_logger(*, verbose: bool = False, colored_logs: bool = False):
    log_level = logging.INFO if not verbose else logging.DEBUG
    log_format = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"

    logging.basicConfig(level=log_level, format=log_format)
    if colored_logs:
        level_styles = deepcopy(coloredlogs.DEFAULT_LEVEL_STYLES)
        level_styles["debug"] = {"color": "white", "faint": True}
        coloredlogs.install(
            fmt=log_format,
            level=log_level,
            field_styles={
                "asctime": {"color": "white", "faint": True},
                "hostname": {"color": "white", "faint": True},
                "levelname": {"color": "white", "faint": True, "bold": True},
                "name": {"color": "white", "faint": True},
                "programname": {"color": "white", "faint": True},
                "username": {"color": "white", "faint": True},
            },
            level_styles=level_styles
            # isatty=False,
        )

    logging.getLogger("sh.command").setLevel(logging.WARNING)
    logging.getLogger("sh.stream_bufferer").setLevel(logging.WARNING)
    logging.getLogger("sh.streamreader").setLevel(logging.WARNING)
    logging.getLogger("docker.api.build").setLevel(logging.WARNING)
    logging.getLogger("docker.auth").setLevel(logging.WARNING)
    logging.getLogger("docker.utils.config").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def dump_loggers_handlers(tag: Optional[str] = None):
    print(f"============= START loggers dump {tag or ''} ==================")
    loggers = {"root": logging.getLogger(), **logging.Logger.manager.loggerDict}
    for idx, (name, logger) in enumerate(loggers.items()):
        handler, other_logger = coloredlogs.find_handler(logger, coloredlogs.match_stream_handler)
        print(idx, name, logger, "->", other_logger, handler)
    print(f"============= STOP loggers dump {tag or ''} ==================")


def log_dict(title: str, dict_: Dict[str, Any]):
    LOGGER = logging.getLogger(__name__)
    LOGGER.info(title)
    for key, value in dict_.items():
        LOGGER.info(f"\t{key} = {value}")


def print_dict(title: str, dict_: Dict[str, Any]):
    print(title)
    for key, value in dict_.items():
        print(f"\t{key} = {value}")


def dump_sh_logs(name, logs, limit: Optional[int] = None):
    LOGGER = logging.getLogger(__name__)
    lines = logs.decode("utf-8").split("\n")
    if lines:
        LOGGER.warning(name)
        if limit is not None:
            lines = lines[-limit:]
        for line in lines:
            print("\t" + line, flush=True)
