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
import pathlib
import shutil
import zipfile

import yaml

from model_navigator.exceptions import ModelNavigatorInvalidPackageException

LOGGER = logging.getLogger(__name__)


class TritonPackage:
    def __init__(self, path):
        self.path = path

        # note: the zipfile never gets closed. We assume only a few such
        # objects will be created
        self.arc = zipfile.ZipFile(self.path, "r")

    def open(self, fpath):
        return self.arc.open(fpath, "r")

    @property
    def model_metrics_path(self):
        return "results/metrics-model-inference.csv"

    @property
    def gpu_metrics_path(self):
        return "results/metrics-gpu-inference.csv"

    def status(self):
        try:
            with self.arc.open("status.yaml") as f:
                status = yaml.safe_load(f.read())
            return status
        except KeyError as e:
            raise ModelNavigatorInvalidPackageException(str(e))

    def copy_model_to_repository(self, model_config, output_repository, output_name=None, overwrite=False):
        """Copy a model out of the package into a valid model repository directory structure"""
        if output_name is None:
            output_name = model_config

        LOGGER.info("Copying model configuration %s to %s/%s", model_config, output_repository, output_name)

        def _extract(pkg_path, out_path):
            # pytype: disable=attribute-error
            relname = pathlib.Path(pkg_path.at).relative_to(pathlib.Path("model-store")).as_posix()
            # pytype: enable=attribute-error
            if relname in symlinks:
                return _extract(zipfile.Path(self.arc) / "model-store" / symlinks[relname], out_path)
            LOGGER.debug("Extracting %s to %s", pkg_path, out_path)
            if pkg_path.is_dir():
                out_path.mkdir(parents=True, exist_ok=overwrite)
                for chld in pkg_path.iterdir():
                    _extract(chld, out_path / chld.name)
                return

            with out_path.open(out_mode) as f, pkg_path.open() as g:
                shutil.copyfileobj(g, f)

        try:
            path_in_pkg = zipfile.Path(self.arc) / "model-store" / model_config
            output_model_dir = output_repository / output_name
            symlinks = yaml.load(self.arc.read("model-store/symlinks.yaml"), Loader=yaml.SafeLoader)
            out_mode = "wb" if overwrite else "xb"

            _extract(path_in_pkg, output_model_dir)
        except KeyError as e:
            raise ModelNavigatorInvalidPackageException(str(e))
