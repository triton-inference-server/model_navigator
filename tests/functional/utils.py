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
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


def load_file(file, label, target):
    if not os.path.isfile(file):
        raise ValueError(f"Provided file {file} for {target} does not exists")

    spec = importlib.util.spec_from_file_location(name=label, location=file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pytype: disable=attribute-error
    return getattr(module, target)


def download_model_and_run_script(config_path: Path, script_with_args: List[str], downloader_path: str):
    import sh

    config = load_config(config_path)
    src_model_path = Path(config["model_path"])
    _download_model(config_path=config_path, output_model_path=src_model_path, downloader_path=downloader_path)
    cmd_name, *args = script_with_args
    LOGGER.debug(f"Run {cmd_name} {' '.join(args)}")
    cmd = sh.Command(cmd_name)
    cmd(args, _in=sys.stdin, _out=sys.stdout, _err=sys.stderr)


def expand_model_path(config):
    # expand model_path with environment variables
    src_model_path = Path(os.path.expandvars(config["model_path"]))
    if "$" in src_model_path.as_posix():
        raise ValueError(f"Could not expend vars for {src_model_path}")
    # update and save model-navigator config file
    config["model_path"] = src_model_path.as_posix()
    LOGGER.debug(f"Expanded model_path: {src_model_path}")
    return config


def save_config(config: Dict[str, Any], updated_config_path: Path):
    import yaml

    updated_config_path.parent.mkdir(parents=True, exist_ok=True)
    with updated_config_path.open("w") as config_file:
        yaml.dump(config, config_file)


def load_config(config_path):
    import yaml

    with config_path.open("r") as config_file:
        return yaml.safe_load(config_file)


def _download_model(config_path: Path, output_model_path: Path, downloader_path: str):
    config = load_config(config_path)

    downloader_cls = load_file(downloader_path, "downloader_cls", "downloader_cls")
    downloader = downloader_cls(config)
    LOGGER.debug(f"Using downloader {downloader_cls}")

    workdir = Path(__file__).parent.parent.parent.resolve()

    from model_navigator.utils.docker import DockerImage

    docker_image = DockerImage(f"{downloader.framework.image}:{config['container_version']}-{downloader.framework.tag}")
    download_model_path = "tests/functional/download_model.py"
    dataloader_kwargs = config["model_downloader_kwargs"]

    docker_container = None
    try:
        docker_container = docker_image.run_container(
            workdir_path=workdir,
            mount_as_volumes=[
                output_model_path.parent.resolve(),
                config_path.parent.resolve(),
                Path(__file__).parent.resolve(),
                workdir,
                *(Path(p) for p in dataloader_kwargs.get("mounts", [])),
                *(Path(p) for p in downloader.mounts),
            ],
            environment={"PYTHONPATH": workdir.as_posix(), **dataloader_kwargs.get("envs", {}), **downloader.envs},
        )

        docker_container.run_cmd(
            (
                "bash -c '"
                f"python {download_model_path} -vvvv "
                f"--downloader-path {downloader_path} "
                f"--config-path {config_path.resolve()} "
                f"--output-path {output_model_path.resolve()}"
                "'"
            ),
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )
        LOGGER.debug("Finished download_model.py script")
    finally:
        if docker_container:
            docker_container.kill()


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
