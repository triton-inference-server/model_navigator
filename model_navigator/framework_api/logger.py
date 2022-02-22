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
from pathlib import Path

import coloredlogs

LOGGER = logging.getLogger("Navigator API")
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


def add_log_file_handler(log_dir: Path):
    log_file = log_dir / "navigator.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)
