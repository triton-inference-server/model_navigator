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
import sys
from pathlib import Path
from typing import Any, Dict, List

from tests.utils.common import load_config
from tests.utils.downloader import DOWNLOADERS, DownloaderConfig

LOGGER = logging.getLogger(__name__)


def resolve_config_paths(config_paths):
    CONFIG_SUFFIX = ".yaml"
    resolved_config_paths = []
    for entry in config_paths:
        entry = Path(entry)
        if "*" in entry.name:
            resolved_config_paths.extend(list(entry.parent.glob(entry.name)))
        elif entry.is_dir():
            resolved_config_paths.extend(list(entry.glob("*")))
        else:
            resolved_config_paths.append(entry)
    resolved_config_paths = [p for p in resolved_config_paths if p.suffix == CONFIG_SUFFIX]
    return sorted(set(resolved_config_paths))


def expand_model_path(config):
    # expand model_path with environment variables
    src_model_path = Path(os.path.expandvars(config["model_path"]))
    if "$" in src_model_path.as_posix():
        raise ValueError(f"Could not expend vars for {src_model_path}")
    # update and save model-navigator config file
    config["model_path"] = src_model_path.as_posix()
    LOGGER.debug(f"Expanded model_path: {src_model_path}")
    return config


def download_model_and_run_script(config_path: Path, script_with_args: List[str]):
    import sh

    test_config = load_config(config_path)
    downloader_config = DownloaderConfig.from_test_config(test_config)
    src_model_path = Path(test_config["model_path"])

    DownloaderCls = DOWNLOADERS[downloader_config.model_url.scheme]
    downloader = DownloaderCls(config_path)
    downloader.download_model(downloader_config=downloader_config, output_path=src_model_path)
    cmd_name, *args = script_with_args
    LOGGER.debug(f"Run {cmd_name} {' '.join(args)}")
    cmd = sh.Command(cmd_name)
    cmd(args, _in=sys.stdin, _out=sys.stdout, _err=sys.stderr)


def save_config(config: Dict[str, Any], updated_config_path: Path):
    import yaml

    updated_config_path.parent.mkdir(parents=True, exist_ok=True)
    with updated_config_path.open("w") as config_file:
        yaml.dump(config, config_file)


def run_test(config_path, work_dir, config_suffix, get_cmd_fn):
    def _rewrite_config_file(work_dir: Path, config_path: Path, suffix: str):
        os.environ["TEST_TEMP_DIR"] = work_dir.as_posix()
        config = load_config(config_path)
        config = expand_model_path(config)
        updated_config_path = Path(config["model_path"]).with_suffix(suffix)
        LOGGER.debug(f"Saving updated config in {updated_config_path}")
        save_config(config, updated_config_path)
        return updated_config_path

    work_dir = Path(work_dir)
    updated_config_path = _rewrite_config_file(work_dir, config_path, config_suffix)
    cmd = get_cmd_fn(workdir=work_dir, config_path=updated_config_path)
    download_model_and_run_script(
        config_path=updated_config_path,
        script_with_args=cmd,
    )
