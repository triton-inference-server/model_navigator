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

from jinja2 import Environment, FileSystemLoader

from model_navigator.framework import Framework
from model_navigator.kubernetes import internals, utils
from model_navigator.kubernetes.evaluator import EvaluatorChartCreator
from model_navigator.kubernetes.inference import InferenceChartCreator
from model_navigator.utils.source import navigator_install_url


class Generator:
    def __call__(
        self,
        *,
        chart_name: str,
        output_path: pathlib.Path,
        container_version: str,
        framework,
        navigator_cmds: typing.List[str],
        evaluator_cmds: typing.List[str],
        create_evaluator: bool = True,
        create_dockerfile: bool = False,
    ):
        self.catalog = output_path
        self._create_catalog()

        chart_basename = chart_name.lower().replace("_", "-")
        inference_chart_creator = InferenceChartCreator(
            chart_name=f"{chart_basename}-{InferenceChartCreator.NAME}",
            catalog=self.catalog,
            container_version=container_version,
            framework=framework,
            cmds=navigator_cmds,
        )
        inference_chart_creator.create()

        evaluator_chart_creator = None
        if create_evaluator:
            evaluator_chart_creator = EvaluatorChartCreator(
                chart_name=f"{chart_basename}-{EvaluatorChartCreator.NAME}",
                catalog=self.catalog,
                container_version=container_version,
                framework=framework,
                cmds=evaluator_cmds,
            )
            evaluator_chart_creator.create()

        if create_dockerfile:
            self._create_dockerfile(
                container_version=container_version,
                framework=framework,
                inference_chart_creator=inference_chart_creator,
                evaluator_chart_creator=evaluator_chart_creator,
            )

    def _create_catalog(self):
        if self.catalog.is_dir():
            shutil.rmtree(self.catalog.as_posix())

        self.catalog.mkdir(parents=True, exist_ok=True)

    def _create_dockerfile(
        self,
        container_version: str,
        framework: Framework,
        inference_chart_creator: InferenceChartCreator,
        evaluator_chart_creator: EvaluatorChartCreator,
    ):
        docker_template = "Dockerfile.jinja2"
        template_path = internals.package_dir / "templates"
        env = Environment(loader=FileSystemLoader(template_path.as_posix()))

        install_url = navigator_install_url(framework)

        config_local_path = "config.yaml"
        tags = {
            "FROM_IMAGE_NAME": framework.container_image(version=container_version),
            "INSTALL_URL": install_url,
            "DEPLOYER_LOCAL": inference_chart_creator.entrypoint_local_path,
            "DEPLOYER_DOCKER": inference_chart_creator.entrypoint_docker_path,
            "EVALUATOR_LOCAL": evaluator_chart_creator.entrypoint_local_path,
            "EVALUATOR_DOCKER": evaluator_chart_creator.entrypoint_docker_path,
            "CONFIG_LOCAL": config_local_path,
            "CONFIG_DOCKER": internals.Paths.CONFIG_PATH,
        }

        template = env.get_template(docker_template)

        dockerfile = self.catalog / "Dockerfile"
        with open(dockerfile, "w") as fh:
            fh.write(template.render(**tags))

        utils.append_copyright(filename=dockerfile, tag="#")


generator = Generator()
