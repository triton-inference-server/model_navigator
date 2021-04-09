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
import typing

import os
import pathlib
import shutil
import sys

from jinja2 import Environment, FileSystemLoader

from ..framework import Framework, PyTorch
from . import internals, utils
from .inference import InferenceChartCreator
from .tester import TesterChartCreator


class Generator:
    def __call__(
        self,
        output_path: pathlib.Path,
        container_version: str,
        framework: Framework,
        deployer_parameters: typing.NamedTuple,
        deployer_cmds: typing.List[str],
        tester_parameters: typing.NamedTuple,
        tester_cmds: typing.List[str],
        create_tester: bool = True,
        create_dockerfile: bool = False,
    ):
        self.catalog = output_path
        self._create_catalog()

        if not hasattr(deployer_parameters, "model_name"):
            raise ValueError("Model name parameters is required.")

        inference_chart_creator = self._create_inference_chart(
            container_version=container_version,
            framework=framework,
            cmds=deployer_cmds,
            parameters=deployer_parameters,
        )

        tester_chart_creator = None
        if create_tester:
            tester_chart_creator = self._create_tester_chart(
                container_version=container_version, framework=framework, cmds=tester_cmds, parameters=tester_parameters
            )

        if create_dockerfile:
            self._create_dockerfile(
                container_version=container_version,
                framework=framework,
                inference_chart_creator=inference_chart_creator,
                tester_chart_creator=tester_chart_creator,
            )

    def _create_catalog(self):
        if self.catalog.is_dir():
            shutil.rmtree(self.catalog.as_posix())

        self.catalog.mkdir()

    def _create_inference_chart(
        self,
        container_version: str,
        framework: Framework,
        cmds: typing.List[str],
        parameters: typing.NamedTuple,
    ):
        chart = InferenceChartCreator(
            catalog=self.catalog,
            container_version=container_version,
            framework=framework,
            cmds=cmds,
            parameters=parameters,
        )
        chart.create()

        return chart

    def _create_tester_chart(
        self,
        container_version: str,
        framework: Framework,
        cmds: typing.List[str],
        parameters: typing.NamedTuple,
    ):
        chart = TesterChartCreator(
            catalog=self.catalog,
            container_version=container_version,
            framework=framework,
            cmds=cmds,
            parameters=parameters,
        )
        chart.create()

        return chart

    def _create_dockerfile(
        self,
        container_version: str,
        framework: Framework,
        inference_chart_creator: InferenceChartCreator,
        tester_chart_creator: TesterChartCreator,
    ):
        docker_template = "Dockerfile.jinja2"
        template_path = internals.package_dir / "templates"
        env = Environment(loader=FileSystemLoader(template_path))

        extras = "tf"
        if framework == PyTorch:
            extras = "pyt"

        # The package importlib_metadata is in a different place, depending on the python version.
        if sys.version_info < (3, 8):
            import importlib_metadata
        else:
            import importlib.metadata as importlib_metadata

        metadata = importlib_metadata.metadata("model_navigator")
        version = metadata["Version"]
        url = metadata["Home-page"]
        install_url = f"git+{url}.git@v{version}#egg=model_navigator[{extras}]"

        tags = {
            "FROM_IMAGE_NAME": framework.container_image(version=container_version),
            "INSTALL_URL": install_url,
            "DEPLOYER_LOCAL": inference_chart_creator.entrypoint_local_path,
            "DEPLOYER_DOCKER": inference_chart_creator.entrypoint_docker_path,
            "TESTER_LOCAL": tester_chart_creator.entrypoint_local_path,
            "TESTER_DOCKER": tester_chart_creator.entrypoint_docker_path,
        }

        template = env.get_template(docker_template)

        dockerfile = self.catalog / "Dockerfile"
        with open(dockerfile, "w") as fh:
            fh.write(template.render(**tags))

        utils.append_copyright(filename=dockerfile, tag="#")


generator = Generator()
