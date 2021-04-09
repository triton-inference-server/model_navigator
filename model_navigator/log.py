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
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from model_navigator.config import ModelNavigatorBaseConfig


def set_tf_verbosity(verbose: bool = False):
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    tf_cpp_min_log_level = 0 if verbose else 1
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(tf_cpp_min_log_level)
    os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"


def set_logger(*, verbose: bool = False):
    log_format = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    log_level = logging.INFO if not verbose else logging.DEBUG
    logging.basicConfig(level=log_level, format=log_format)
    logging.getLogger("sh.command").setLevel(logging.WARNING)
    logging.getLogger("sh.stream_bufferer").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def is_root_logger_verbose() -> bool:
    return logging.getLogger().getEffectiveLevel() == logging.DEBUG


def log_dict(title: str, dict_: Dict[str, Any]):
    LOGGER = logging.getLogger(__name__)
    LOGGER.info(title)
    for key, value in dict_.items():
        LOGGER.info(f"\t{key} = {value}")


def dump_sh_logs(name, logs, limit: Optional[int] = None):
    LOGGER = logging.getLogger(__name__)
    lines = logs.decode("utf-8").split("\n")
    if lines:
        LOGGER.warning(name)
        if limit is not None:
            lines = lines[-limit:]
        for line in lines:
            print("\t" + line, flush=True)


def section_header(section_name: str) -> str:
    return f"\n\n================== {section_name} ==================\n\n"


class FileLogger:
    def __init__(self, config: ModelNavigatorBaseConfig, name: str):
        filename = f"{name}.log"
        self.file_path = self.get_logs_dir(config) / filename

    def log(self, content: str):
        with open(self.file_path, "a+") as f:
            f.write(content)

    @classmethod
    def get_logs_dir(cls, config: ModelNavigatorBaseConfig):
        return Path(config.workspace_path) / "logs"
