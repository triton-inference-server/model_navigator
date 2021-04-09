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
from typing import Dict

import re
from pathlib import Path

import yaml


class CustomDumper(yaml.Dumper):
    def ignore_aliases(self, data: Dict) -> bool:
        return True


class YAMLGenerator:
    def __call__(self, file: Path, data: Dict):
        content = yaml.dump(data, Dumper=CustomDumper, width=120, sort_keys=False, default_flow_style=False)
        content = self._post_process(content=content)
        with open(file, "w") as f:
            f.write(content)

    def _post_process(self, content: str) -> str:
        content = content.replace("'", "")
        content = content.replace("}}: null", "}}")
        content = content.replace("- {{- if", "{{- if")
        content = content.replace("- {{ end }}", "{{ end }}")
        content = content.replace("- {{ else }}", "{{ else }}")
        content = re.sub("{{cond[0-9]+}}", "", content)
        return content


generator = YAMLGenerator()
