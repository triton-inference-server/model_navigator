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
import zipfile

import yaml

from model_navigator import LOGGER
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.results import ResultsStore, State
from model_navigator.utils import Workspace, device
from model_navigator.utils.config import dataclass2dict

FORMAT_VERSION = "0.0.1"


def pack_workspace(
    workspace: Workspace,
    package_path: pathlib.Path,
    navigator_config,
):
    LOGGER.info(f"Creating package from workspace {workspace.path} to {package_path}")

    results_store = ResultsStore(workspace)

    LOGGER.debug("Collecting conversion information.")
    conversion_results = results_store.load("convert_model")

    LOGGER.debug("Collecting Triton configuration results.")
    configuration_results = results_store.load("configure_models_on_triton")

    conversion_log_path = "conversion-logs"
    conversion_logs = []

    configurator_log_path = "configurator-logs"
    configurator_logs = []

    LOGGER.debug("Creating package content.")
    models = []
    for conversion_result in conversion_results:
        model_data = {
            "status": conversion_result.status.state.value,
            "name": conversion_result.output_model.name if conversion_result.output_model else None,
            "log_file": None,
            "conversion_config": dataclass2dict(conversion_result.conversion_config),
        }

        if conversion_result.status.state != State.SUCCEEDED or not conversion_result.output_model:
            log_path = conversion_result.status.log_path
            if log_path:
                log_path = pathlib.Path(log_path)
                model_data["log_file"] = pathlib.Path(conversion_log_path) / log_path.name
                conversion_logs.append(log_path)

            model_data["model_stores"] = []
        else:
            model_stores = []
            for config_result in filter(
                lambda c: c.model.name == conversion_result.output_model.name, configuration_results
            ):
                model_store_data = {
                    "status": config_result.status.state.value,
                    "name": config_result.model_config_name,
                    "log_file": None,
                    "optimizations": dataclass2dict(config_result.optimization_config),
                }
                if config_result.status.state != State.SUCCEEDED:
                    log_path = config_result.status.log_path
                    if log_path:
                        log_path = pathlib.Path(log_path)
                        model_store_data["log_file"] = pathlib.Path(configurator_log_path) / log_path.name
                        configurator_logs.append(log_path)

                model_stores.append(model_store_data)

            model_data["model_stores"] = model_stores

        models.append(model_data)

    status = {
        "format_version": FORMAT_VERSION,
        "config": navigator_config,
        "models": models,
        "environment": device.get_environment_info(),
    }

    LOGGER.debug("Compressing package to single file.")
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_STORED) as package:
        results_path = pathlib.Path(workspace.path / "analyzer" / "results")
        for result in results_path.iterdir():
            package.write(result, arcname=pathlib.Path("results") / result.relative_to(results_path))

        checkpoints_path = pathlib.Path(workspace.path / "analyzer" / "checkpoints")
        for checkpoint in checkpoints_path.iterdir():
            package.write(checkpoint, arcname=pathlib.Path("checkpoints") / checkpoint.name)

        # zipfile does not support symlinks, so we store symlink info in a yaml file
        symlinks = {}
        models_path = pathlib.Path(workspace.path / "analyzer" / "model-store")
        for model in models_path.glob("**/*"):
            relname = model.relative_to(models_path)
            outname = pathlib.Path("model-store") / relname
            if model.is_symlink():
                symlinks[relname.as_posix()] = (
                    pathlib.Path(os.path.realpath(model.as_posix())).relative_to(models_path).as_posix()
                )
                package.writestr(outname.as_posix(), "")
                continue
            package.write(model, arcname=outname)
        package.writestr("model-store/symlinks.yaml", yaml.dump(symlinks, width=240, sort_keys=False))

        for log_file in conversion_logs:
            package.write(log_file, arcname=pathlib.Path(conversion_log_path) / log_file.name)

        for log_file in configurator_logs:
            package.write(log_file, arcname=pathlib.Path(configurator_log_path) / log_file.name)

        package.writestr("status.yaml", yaml.dump(status, width=240, sort_keys=False))

    if not package_path.is_file():
        raise ModelNavigatorException(f"Package not found in {package_path}.")

    LOGGER.info(f"Package stored in {package_path}.")


if __name__ == "__main__":
    pack_workspace(Workspace("navigator_workspace"), pathlib.Path("test.triton.nav"), {})
