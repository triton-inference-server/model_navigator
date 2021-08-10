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
import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pytest
from click.testing import CliRunner


@dataclass
class RunContext:
    cwd: Optional[str] = None
    cmd: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None
    return_code: Optional[int] = None
    output: Optional[str] = None


@dataclass
class ModelConfigPair:
    model_path: Path
    config_path: Path


class ArtifactoryRepository:
    CATALOG = {
        ("TorchScript/simple", "simple"): ModelConfigPair(
            model_path=Path("tests/files/models/identity.scripted.pt"),
            config_path=Path("tests/files/models/identity.scripted.pt.simple.nav.yaml"),
        ),
    }

    def copy_model_and_config(self, *, model_type: str, config_type: str, to: Path):
        to.mkdir(parents=True, exist_ok=True)
        pair = self.CATALOG[(model_type, config_type)]
        for src_path in dataclasses.astuple(pair):
            dst_path = to / src_path.name
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        return to / pair.config_path.name


@pytest.fixture(scope="function")
def runner(request):
    return CliRunner()


@pytest.fixture(scope="function")
def run_context(request):
    return RunContext(parameters={})


@pytest.fixture(scope="session")
def artifactory_repo(request):
    return ArtifactoryRepository()
