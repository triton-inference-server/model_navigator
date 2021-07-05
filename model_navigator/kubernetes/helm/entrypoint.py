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
import pathlib
import typing

LOGGER = logging.getLogger(__name__)


class Entrypoint:
    def __init__(self, filename: pathlib.Path, cmds: typing.List[str]):
        self.filename = filename
        self.cmds = cmds

    def create(self):
        init_commands = self._initial_cmds()

        entrypoint_cmds = list()
        entrypoint_cmds.extend(init_commands)

        for cmd in self.cmds:
            entrypoint_cmds.append(cmd)

        content = "\n".join(entrypoint_cmds)
        with open(self.filename, "w") as f:
            f.write(content)

        result = os.system(f'ex +"set syn=sh" +"norm gg=G" -cwq {self.filename}')
        if result != 0:
            LOGGER.error(f"Failed running script formatting. Exit code {result}")
            exit(1)

        result = os.system(f"chmod +x {self.filename}")
        if result != 0:
            LOGGER.error(f"Failed running script changing script mode. Exit code {result}")
            exit(1)

    def _initial_cmds(self):
        cmds = [
            "#!/bin/bash",
            "set -xe",
            "export PYTHONUNBUFFERED=1",
            "export PYTHONPATH=`pwd`",
        ]

        return cmds
