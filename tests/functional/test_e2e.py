#!/usr/bin/env python3
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
from tempfile import TemporaryDirectory

from model_navigator.log import init_logger
from tests.functional.utils import resolve_config_paths, run_test
from tests.utils.common import load_config

LOGGER = logging.getLogger("test_run")

CONFIG_SUFFIX = ".nav_test_run.yaml"


def _get_cmds(*, workdir: Path, config_path: Path):
    config = load_config(config_path)
    workspace_path = workdir / config["model_name"]
    return [
        "model-navigator",
        "run",
        "--workspace-path",
        workspace_path.as_posix(),
        "--override-workspace",
        "--config-path",
        config_path.as_posix(),
        "-vvv",
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("-v", "--verbose", help="Provides verbose logs", action="count", default=0)
    parser.add_argument("--work-dir", help="Directory where models can be obtained and workspace dirs can be created")
    parser.add_argument("config_paths", nargs="+", help="Path to configuration files")
    args = parser.parse_args()

    init_logger(verbose=args.verbose)

    import sh

    for config_path in resolve_config_paths(args.config_paths):
        LOGGER.info(f"======== Running config: {config_path} =======")

        try:

            if not args.work_dir:
                with TemporaryDirectory() as work_dir:
                    run_test(config_path, work_dir, CONFIG_SUFFIX, get_cmd_fn=_get_cmds)
            else:
                run_test(config_path, args.work_dir, CONFIG_SUFFIX, get_cmd_fn=_get_cmds)
        except sh.ErrorReturnCode as e:
            LOGGER.error(f"Error occurred during running {e.full_cmd}")
            LOGGER.error("--- stdout --- \n" + e.stdout.decode("utf-8"))
            LOGGER.error("--- stderr --- \n" + e.stderr.decode("utf-8"))
        except Exception as e:
            LOGGER.exception(f"Error occurred during running config: {config_path}; {e}")
        LOGGER.info(f"======== Finished config: {config_path} =======")


if __name__ == "__main__":
    main()
