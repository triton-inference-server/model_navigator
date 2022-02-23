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
import shutil
import typing

from jinja2 import Environment, FileSystemLoader

from model_navigator.framework import Framework
from model_navigator.kubernetes.evaluator import EvaluatorChartCreator
from model_navigator.kubernetes.inference import InferenceChartCreator
from model_navigator.kubernetes.internals import Paths, package_dir
from model_navigator.kubernetes.utils import append_copyright
from model_navigator.utils.source import navigator_install_url


def _chart_version_from_docker_tag(docker_tag):
    from semver import VersionInfo
    return docker_tag if docker_tag and VersionInfo.isvalid(docker_tag) else None


class Generator:
    def __call__(
        self,
        *,
        chart_name: str,
        output_path: pathlib.Path,
        chart_version: typing.Optional[str],
        triton_docker_image: str,
        framework_docker_image: str,
        framework,
        navigator_cmds: typing.List[str],
        evaluator_cmds: typing.List[str],
        create_evaluator: bool = True,
        create_dockerfile: bool = False,
    ):
        self.catalog = output_path
        self._create_catalog()

        chart_basename = chart_name.lower().replace("_", "-")

        from docker.utils import parse_repository_tag

        _, triton_docker_tag = parse_repository_tag(triton_docker_image)
        _, framework_docker_tag = parse_repository_tag(framework_docker_image)
        chart_version = (
            chart_version
            or _chart_version_from_docker_tag(triton_docker_tag)
            or _chart_version_from_docker_tag(framework_docker_tag)
            or "0.0.1"
        )
        inference_chart_creator = InferenceChartCreator(
            chart_name=f"{chart_basename}-{InferenceChartCreator.NAME}",
            catalog=self.catalog,
            chart_version=chart_version,
            docker_image=triton_docker_image,
            framework=framework,
            cmds=navigator_cmds,
        )
        inference_chart_creator.create()

        evaluator_chart_creator = None
        if create_evaluator:
            evaluator_chart_creator = EvaluatorChartCreator(
                chart_name=f"{chart_basename}-{EvaluatorChartCreator.NAME}",
                catalog=self.catalog,
                chart_version=chart_version,
                docker_image=None,
                framework=framework,
                cmds=evaluator_cmds,
            )
            evaluator_chart_creator.create()

        if create_dockerfile:
            self._create_dockerfile(
                docker_image=framework_docker_image,
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
        docker_image: str,
        framework: Framework,
        inference_chart_creator: InferenceChartCreator,
        evaluator_chart_creator: EvaluatorChartCreator,
    ):
        docker_template = "Dockerfile.jinja2"
        template_path = package_dir / "templates"
        env = Environment(
            loader=FileSystemLoader(template_path.as_posix()),
            autoescape=True,
        )

        install_url = navigator_install_url(framework, extras=["cloud"])

        config_local_path = "config.yaml"
        tags = {
            "FROM_IMAGE_NAME": docker_image,
            "INSTALL_URL": install_url,
            "DEPLOYER_LOCAL": inference_chart_creator.entrypoint_local_path,
            "DEPLOYER_DOCKER": inference_chart_creator.entrypoint_docker_path,
            "EVALUATOR_LOCAL": evaluator_chart_creator.entrypoint_local_path,
            "EVALUATOR_DOCKER": evaluator_chart_creator.entrypoint_docker_path,
            "CONFIG_LOCAL": config_local_path,
            "CONFIG_DOCKER": Paths.CONFIG_PATH,
        }

        template = env.get_template(docker_template)

        dockerfile = self.catalog / "Dockerfile"
        with open(dockerfile, "w") as fh:
            fh.write(template.render(**tags))

        append_copyright(filename=dockerfile, tag="#")


generator = Generator()
