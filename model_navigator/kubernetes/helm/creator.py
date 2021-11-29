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
import pathlib
import shutil
import typing

from model_navigator import framework
from model_navigator.kubernetes.helm import chart, entrypoint
from model_navigator.kubernetes.internals import package_dir
from model_navigator.kubernetes.utils import append_copyright
from model_navigator.kubernetes.yaml import generator as yaml_generator


def hasclass(name):
    def wrapping(f):
        def wrapper(self, *args, **kwargs):
            if not getattr(self, name):
                return

            return f(self, *args, **kwargs)

        return wrapper

    return wrapping


class ChartCreator:
    # DEFINITION NEEDED
    NAME = None
    DESCRIPTION = None

    ENTRYPOINT_FILE_NAME = None
    DEPLOYMENT_CLS = None
    SERVICE_CLS = None
    VALUES_CLS = None

    CHART_CLS = chart.Chart
    ENTRYPOINT_CLS = entrypoint.Entrypoint

    ENTRYPOINT_FILE_LOCAL = pathlib.Path("toolkit")
    ENTRYPOINT_FILE_DOCKER = pathlib.Path("/opt/workspace")

    def __init__(
        self,
        *,
        catalog: typing.Union[str, pathlib.Path],
        chart_version: str,
        docker_image: typing.Optional[str] = None,
        framework: framework.Framework,
        chart_name: str,
        cmds: typing.List[str],
    ):
        self.catalog = pathlib.Path(catalog)
        self.framework = framework
        self.chart_version = chart_version
        self.docker_image = docker_image
        self.name = chart_name
        self.cmds = cmds

    def create(self):
        self._create_dir()
        self._create_chart_file()
        self._create_values_file()
        self._create_deployment_file()
        self._create_service_file()
        self._create_entrypoint_script()
        self._create_helpers_file()

    @property
    def description(self) -> str:
        return f"{self.DESCRIPTION} for {self.name}"

    @property
    def chart_dir(self) -> pathlib.Path:
        return self.catalog / self.name

    @property
    def template_dir(self) -> pathlib.Path:
        return self.chart_dir / "templates"

    @property
    def entrypoint_local_path(self) -> pathlib.Path:
        return self.ENTRYPOINT_FILE_LOCAL / self.ENTRYPOINT_FILE_NAME

    @property
    def entrypoint_docker_path(self) -> pathlib.Path:
        return self.ENTRYPOINT_FILE_DOCKER / self.ENTRYPOINT_FILE_NAME

    def _create_dir(self):
        if self.chart_dir.is_dir():
            shutil.rmtree(self.chart_dir.as_posix())

        self.chart_dir.mkdir(parents=True)
        self.template_dir.mkdir(parents=True)

    def _create_yaml_file(self, file_path: pathlib.Path, data: typing.Dict):
        yaml_generator(file=file_path, data=data)

    @hasclass("CHART_CLS")
    def _create_chart_file(self):
        chart = self.CHART_CLS(
            name=self.name,
            description=self.description,
            version=self.chart_version,
        )

        chart_file = self.chart_dir / "Chart.yaml"
        self._create_yaml_file(file_path=chart_file, data=chart.data())

        append_copyright(filename=chart_file, tag="#")

    @hasclass("VALUES_CLS")
    def _create_values_file(self):
        values = self.VALUES_CLS(docker_image=self.docker_image)

        values_file = self.chart_dir / "values.yaml"
        self._create_yaml_file(file_path=values_file, data=values.data())

        append_copyright(filename=values_file, tag="#")

    @hasclass("DEPLOYMENT_CLS")
    def _create_deployment_file(self):
        entrypoint_file = self.ENTRYPOINT_FILE_DOCKER / self.ENTRYPOINT_FILE_NAME
        deployment = self.DEPLOYMENT_CLS(name=self.name, framework=self.framework, entrypoint=entrypoint_file)

        deployment_file = self.template_dir / "deployment.yaml"
        self._create_yaml_file(file_path=deployment_file, data=deployment.data())

        append_copyright(filename=deployment_file, tag="#")

    @hasclass("SERVICE_CLS")
    def _create_service_file(self):
        service = self.SERVICE_CLS()
        values_file = self.template_dir / "service.yaml"
        self._create_yaml_file(file_path=values_file, data=service.data())

        append_copyright(filename=values_file, tag="#")

    @hasclass("ENTRYPOINT_CLS")
    def _create_entrypoint_script(self):
        entrypoint_catalog = self.catalog / self.ENTRYPOINT_FILE_LOCAL
        entrypoint_catalog.mkdir(parents=True, exist_ok=True)

        entrypoint_file = entrypoint_catalog / self.ENTRYPOINT_FILE_NAME
        entrypoint_script = entrypoint.Entrypoint(filename=entrypoint_file, cmds=self.cmds)
        entrypoint_script.create()

        append_copyright(filename=entrypoint_file, tag="#")

    def _create_helpers_file(self):
        tpl_file = "_helpers.tpl"
        local_tpl_file_path = package_dir / "templates" / tpl_file
        tpl_file_path = self.chart_dir / "templates" / tpl_file
        shutil.copy(local_tpl_file_path.as_posix(), tpl_file_path.as_posix())

        append_copyright(filename=tpl_file_path, tag="#")
