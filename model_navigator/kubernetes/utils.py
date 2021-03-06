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
import os
from pathlib import Path
from typing import Optional


def format_env(s: str):
    return s.upper()


def format_value(s: str):
    items = s.split("_")
    formatted = items[0].lower()

    for item in items[1:]:
        formatted += item.capitalize()

    return formatted


def env_var(key: str) -> str:
    return f"${{{format_env(key)}}}"


def append_copyright(filename: Path, tag: str, open_tag: Optional[str] = None, close_tag: Optional[str] = None):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    copyright_file_path = os.path.join(local_dir, "templates", "copyright.tpl")

    content = []
    with open(copyright_file_path) as f:
        copyright = f.readlines()
        if open_tag:
            content.append(f"{open_tag}\n")

        for line in copyright:
            content.append(f"{tag} {line}")

        if close_tag:
            content.append(f"{close_tag}\n")

    with open(filename, "r+") as f:
        file_lines = f.readlines()
        content.extend(file_lines)
        content = "".join(content)

        f.seek(0)
        f.write(content)
