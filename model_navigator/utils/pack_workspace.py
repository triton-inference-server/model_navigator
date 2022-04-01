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
import pathlib
import sys
import zipfile

import yaml

from model_navigator import LOGGER
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.utils import device

FORMAT_VERSION = "0.0.1"


def pack_workspace(workspace_path, package_path, navigator_config, input_package=None):
    LOGGER.info(f"Creating package from workspace {workspace_path} to {package_path}")
    LOGGER.debug("Collecting Helm Chart information.")
    with open(workspace_path / "helm-chart-create_results.yaml") as f:
        create_helm_chart_results = yaml.load(f.read(), Loader=yaml.SafeLoader)

    LOGGER.debug("Collecting Model Analyzer results.")
    with open(workspace_path / "analyze_results.yaml") as f:
        analyze_results = yaml.load(f.read(), Loader=yaml.SafeLoader)

    LOGGER.debug("Creating package content.")
    models = {}
    analyzer_csv_reports = set()
    for res in analyze_results:
        analyzer_csv_reports.add(res["metrics_path"])
        analyzer_csv_reports.add(res["results_path"])
        path = pathlib.Path(res["model_repository"]) / res["model_name"]
        models[res["model_name"]] = {
            "path": path,
            "package_path": pathlib.Path("model-store") / path.relative_to(res["model_repository"]),
        }

    helm_charts_status = []
    for res in create_helm_chart_results:
        path = pathlib.Path(res["helm_chart_dir_path"])
        chart_dir = path.parent
        helm_charts_status.append({"path": (pathlib.Path("helm-charts") / path.relative_to(chart_dir)).as_posix()})

    status = {
        "format_version": FORMAT_VERSION,
        "navigator_config": navigator_config,
        "helm_charts": helm_charts_status,
        "triton_models": [
            {
                "name": model,
                "path": info["package_path"].as_posix(),
            }
            for model, info in models.items()
        ],
        "environment": device.get_environment_info(),
    }

    LOGGER.debug("Compressing package to single file.")
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_STORED) as package:
        for helm_chart_res in create_helm_chart_results:
            path = pathlib.Path(helm_chart_res["helm_chart_dir_path"])
            chart_dir = path.parent
            for fs in path.glob("**/*"):
                package.write(fs, arcname=pathlib.Path("helm-charts") / fs.relative_to(chart_dir))

        for _, info in models.items():
            for fs in info["path"].glob("**/*"):
                package.write(fs, arcname=info["package_path"] / fs.relative_to(info["path"]))

        for csv in analyzer_csv_reports:
            fname = pathlib.Path(csv).name
            package.write(csv, arcname=pathlib.Path("analyzer-results") / fname)

        package.writestr("status.yaml", yaml.dump(status))

        if input_package is not None:
            for fs in input_package.glob("**/*"):
                package.write(fs, arcname=pathlib.Path("input.nav") / fs.relative_to(input_package))

    if not package_path.is_file():
        raise ModelNavigatorException(f"Package not found in {package_path}.")

    LOGGER.info(f"Package stored in {package_path}.")


if __name__ == "__main__":
    try:
        input_package = pathlib.Path(sys.argv[1])
    except Exception:
        input_package = None
    pack_workspace(pathlib.Path("navigator_workspace").absolute(), pathlib.Path("test.triton.nav"), {}, input_package)
