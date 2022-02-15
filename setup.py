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
import pathlib
from configparser import ConfigParser

from setuptools import setup

navigator_dir = pathlib.Path(os.path.abspath(__file__)).parent
setup_file = navigator_dir / "setup.cfg"


def _get_info():
    """
    Obtain repository url and commit based on git version from which Navigator is built
    """
    config_parser = ConfigParser()
    config_parser.read(setup_file)
    current_version = config_parser.get("bumpversion", "current_version")
    version = f"v{current_version}"
    repository_url = config_parser.get("metadata", "url")

    try:
        from git import InvalidGitRepositoryError, Repo

        try:
            repo = Repo(navigator_dir.as_posix())
            version = repo.head.commit.hexsha[:8]
            repository_url = repo.remotes.origin.url
        except (InvalidGitRepositoryError, TypeError):
            pass
    except ImportError:
        pass

    return version, repository_url


def write_version():
    """
    Store version and repository url to version file
    """
    import yaml

    version_file = navigator_dir / "model_navigator" / "version.yaml"

    version, repository_url = _get_info()
    data = {"version": version, "repository_url": repository_url}
    with open(version_file, "w") as f:
        yaml.safe_dump(data, f)


# see setup.cfg for config
write_version()
setup()
