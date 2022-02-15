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
import typing

import yaml

from model_navigator.constants import MODEL_NAVIGATOR_DIR
from model_navigator.framework import PyTorch

LOGGER = logging.getLogger(__name__)
version_file = MODEL_NAVIGATOR_DIR / "model_navigator" / "version.yaml"


def navigator_install_url(framework, extras: typing.Optional[typing.List[str]] = None) -> str:
    with open(version_file) as f:
        data = yaml.safe_load(f)

    version = data["version"]
    url = data["repository_url"]

    extras = extras or []
    if framework == PyTorch:
        extras.append("pyt")
    else:
        extras.append("tf")

    install_url = f"git+{url}@{version}#egg=model_navigator[{','.join(extras)}]"

    return install_url


def navigator_is_editable() -> bool:
    dist_files = os.listdir(MODEL_NAVIGATOR_DIR.as_posix())
    editable = "setup.py" in dist_files
    LOGGER.debug(f"Model Navigator in editable mode: {editable}")
    LOGGER.debug(f"Model Navigator path: {MODEL_NAVIGATOR_DIR}")

    return editable
