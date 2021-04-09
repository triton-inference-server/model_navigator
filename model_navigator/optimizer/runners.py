#  Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import IO, List, Sequence

import pickle
import shutil
from pathlib import Path

from docker.types import DeviceRequest
from model_navigator.config import ModelNavigatorBaseConfig
from model_navigator.device import utils
from model_navigator.log import is_root_logger_verbose
from model_navigator.model import InputModel
from model_navigator.optimizer.docker import Docker
from model_navigator.optimizer.transformers import BaseModelTransformer


class PipelineRunInDocker:
    def __init__(
        self,
        pipeline,
        *,
        framework,
        config: ModelNavigatorBaseConfig,
        logs_writer: IO,
        workdir: Path,
    ):
        self._pipeline = pipeline
        self._framework = framework
        self._config = config
        self._logs_writer = logs_writer
        self._verbose = is_root_logger_verbose()
        self._workdir = workdir

    # pytype: disable=bad-return-type
    def execute(self, src_model: InputModel, config: ModelNavigatorBaseConfig) -> List[InputModel]:
        shutil.rmtree(self._pipeline.export_dir, ignore_errors=True)
        self._pipeline.export_dir.mkdir(parents=True, exist_ok=True)

        model_navigator_dir = Path(__file__).parent.parent.parent

        local_paths_to_mount = {
            src_model.path.parent.resolve(),
            self._pipeline.export_dir.resolve(),
            self._workdir.resolve(),
        }
        job_spec_path = self._pipeline.export_dir / ".spec.pkl"
        results_path = self._pipeline.export_dir / ".results.pkl"
        try:
            job_spec_path.parent.mkdir(parents=True, exist_ok=True)
            job_spec = (self._pipeline, src_model, config)
            with job_spec_path.open("wb") as job_spec_file:
                pickle.dump(job_spec, job_spec_file)

            docker = Docker(model_navigator_dir / "model_navigator/optimizer/Dockerfile")
            cmd = f"stdbuf -oL python3 -m model_navigator.optimizer.entrypoint {job_spec_path} {results_path}"
            if self._verbose:
                cmd += " --verbose"

            gpus = utils.get_gpus(self._config)
            devices = [DeviceRequest(device_ids=[gpus[0]], capabilities=[["gpu"]])]
            container_version = self._container_version()
            framework_docker_image = self._framework.container_image(container_version)

            docker.run(
                cmd=cmd,
                devices=devices,
                image_name=framework_docker_image,
                mount_as_volumes=local_paths_to_mount,
                log_writer=self._logs_writer,
                workdir=self._workdir,
                verbose=self._verbose,
            )
            with results_path.open("rb") as results_file:
                results = pickle.load(results_file)
                return results

        finally:
            if job_spec_path.exists():
                job_spec_path.unlink()
            if results_path.exists():
                results_path.unlink()

        # pytype: enable=bad-return-type

    def _container_version(self):
        chunks = self._config.triton_version.split("-")
        version = "-".join(chunks[: len(chunks) - 1])
        return version


class LocalTransformsExecutor:
    def execute(self, transformers: Sequence[BaseModelTransformer], src_model: InputModel) -> Sequence[InputModel]:

        self._clean_export_dirs(transformers)

        result_models = []
        for transformer in transformers:
            # return only not None results of transform tree leaves
            model = transformer.run(src_model)
            if model:
                result_models.append(model)
        return result_models

    def _clean_export_dirs(self, transformers):
        export_dirs = sorted({t.export_dir for t in transformers if t.export_dir})
        for export_dir in export_dirs:
            shutil.rmtree(export_dir, ignore_errors=True)
