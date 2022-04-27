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
import shutil
import sys
from distutils.version import LooseVersion
from pathlib import Path
from typing import List, Optional

import yaml

from model_navigator.converter import DatasetProfileConfig
from model_navigator.exceptions import ModelNavigatorProfileException
from model_navigator.kubernetes.yaml import CustomDumper
from model_navigator.model_analyzer import ModelAnalyzer, ModelAnalyzerProfileConfig
from model_navigator.model_analyzer.config import BaseConfigGenerator, ModelAnalyzerTritonConfig
from model_navigator.model_analyzer.model_analyzer import ModelAnalyzerMode
from model_navigator.model_analyzer.model_analyzer_config import ModelAnalyzerConfig
from model_navigator.perf_analyzer import PerfMeasurementConfig
from model_navigator.triton import DeviceKind
from model_navigator.triton.model_config import TritonModelConfigGenerator
from model_navigator.triton.utils import get_shape_params
from model_navigator.utils import Workspace

LOGGER = logging.getLogger(__name__)

if LooseVersion(sys.version) >= LooseVersion("3.8.0"):
    from importlib.metadata import version

    TRITON_MODEL_ANALYZER_VERSION = LooseVersion(version("triton-model-analyzer"))
else:
    import pkg_resources

    TRITON_MODEL_ANALYZER_VERSION = LooseVersion(pkg_resources.get_distribution("triton-model-analyzer").version)


class Profiler:
    def __init__(
        self,
        *,
        workspace: Workspace,
        triton_docker_image: str,
        gpus: List[str],
        verbose: bool = False,
        profile_config: ModelAnalyzerProfileConfig,
        triton_config: ModelAnalyzerTritonConfig,
        perf_measurement_config: PerfMeasurementConfig,
        dataset_profile_config: Optional[DatasetProfileConfig] = None,
        profiling_data_path: Optional[Path] = None,
    ):
        self._workspace = workspace

        self._triton_config = triton_config
        self._triton_docker_image = triton_docker_image
        self._profile_config = profile_config
        self._dataset_profile_config = dataset_profile_config
        self._profiling_data_path = profiling_data_path
        self._perf_measurement_config = perf_measurement_config

        self._config_generator: ProfileConfigGenerator = ProfileConfigGenerator(
            workspace=self._workspace,
            profile_config=self._profile_config,
            triton_config=triton_config,
            triton_docker_image=triton_docker_image,
            verbose=verbose,
            dataset_profile_config=dataset_profile_config,
            profiling_data_path=profiling_data_path,
            perf_measurement_config=perf_measurement_config,
            gpus=gpus,
        )

        self._profile_config_path = self._config_generator.analyzer_path / "config-profile.yaml"

        self._verbose = verbose
        self._prepare_catalogs()

    def run(self) -> Path:
        config = self._config_generator.generate_config()
        self._profile_config_path.parent.mkdir(parents=True, exist_ok=True)
        with self._profile_config_path.open("w") as config_file:
            config_content = yaml.dump(config, Dumper=CustomDumper)
            LOGGER.debug("Triton Model Analyzer profile config:\n" f"{config_content}")
            config_file.write(config_content)

        analyzer_config = ModelAnalyzerConfig()
        analyzer_config["config-file"] = self._profile_config_path.as_posix()

        analyzer = ModelAnalyzer(config=analyzer_config)
        analyzer.run(mode=ModelAnalyzerMode.PROFILE, verbose=self._verbose)

        latest_checkpoint_path = self._find_latest_checkpoint()

        LOGGER.info(f"Triton Model Analyzer profiling done. Results are stored in {latest_checkpoint_path}")
        return latest_checkpoint_path

    def _find_latest_checkpoint(self):
        checkpoints_paths = sorted(
            self._config_generator.checkpoints_dir_path.glob("*.ckpt"),
            key=lambda path: int(path.stem),
        )
        latest_checkpoint_path = checkpoints_paths[-1] if checkpoints_paths else None
        return latest_checkpoint_path

    def _prepare_catalogs(self):
        def _remove_and_create_dir(dir_path: Path):
            if dir_path.is_dir():
                LOGGER.debug(f"Removing {dir_path}")
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True)

        _remove_and_create_dir(self._config_generator.analyzer_path)


class ProfileConfigGenerator(BaseConfigGenerator):
    def __init__(
        self,
        *,
        workspace: Workspace,
        profile_config: ModelAnalyzerProfileConfig,
        triton_config: ModelAnalyzerTritonConfig,
        perf_measurement_config: PerfMeasurementConfig,
        gpus: List[str],
        triton_docker_image: Optional[str] = None,
        verbose: int = 0,
        dataset_profile_config: Optional[DatasetProfileConfig] = None,
        profiling_data_path: Optional[Path] = None,
    ):
        super().__init__(workspace=workspace, verbose=verbose)
        self._analyzer_triton_log_path = self._analyzer_path / "triton.log"

        self._triton_config = triton_config
        self._triton_docker_image = triton_docker_image
        self._verbose = verbose
        self._profile_config = profile_config
        self._dataset_profile_config = dataset_profile_config
        self._profiling_data_path = profiling_data_path
        self._perf_measurement_config = perf_measurement_config
        self._gpus = gpus

    @property
    def triton_log_path(self) -> Path:
        return self._analyzer_triton_log_path.resolve()

    def generate_config(self):
        model_repository = self._triton_config.model_repository
        models_list = [model_dir.name for model_dir in model_repository.glob("*") if model_dir.is_dir()]
        LOGGER.info(f"Prepare profiling for {len(models_list)} models from {model_repository}:")
        for model_name in models_list:
            LOGGER.info(f"\t- {model_name}")

        model_names_with_profile_config = {
            model_name: self._get_profile_config_for_model(model_name) for model_name in models_list
        }
        if any(profile_config for model_name, profile_config in model_names_with_profile_config.items()):
            models_list = model_names_with_profile_config

        manual_config_search = all(
            isinstance(models_list, dict) and models_list[model_name].get("model_config_parameters")
            for model_name in models_list
        )

        # https://github.com/triton-inference-server/model_analyzer/blob/r22.02/docs/config.md
        config = {
            "run_config_search_disable": manual_config_search,
            "profile_models": models_list,
            "triton_docker_image": self._triton_docker_image,
            "triton_launch_mode": self._triton_config.triton_launch_mode.value,
            "model_repository": model_repository.resolve().as_posix(),
            "checkpoint_directory": self._analyzer_checkpoints_dir_path.as_posix(),
            "output_model_repository_path": self.output_model_repository_path.as_posix(),
            "export_path": self._analyzer_path.resolve().as_posix(),
            "triton_server_flags": {"strict-model-config": False},
            "run_config_search_max_concurrency": self._profile_config.config_search_max_concurrency,
            "run_config_search_max_instance_count": self._profile_config.config_search_max_instance_count,
            "perf_analyzer_timeout": self._perf_measurement_config.perf_analyzer_timeout,
            "perf_analyzer_flags": self._get_perf_analyzer_flags(),
            "triton_server_path": self._triton_config.triton_server_path,
            "override_output_model_repository": True,
            "gpus": list(self._gpus),
            "summarize": self._verbose,
            "verbose": self._verbose,
            "perf_output": self._verbose,
            "triton_output_path": self.triton_log_path.as_posix(),
        }

        if self._perf_measurement_config.perf_analyzer_path:
            config["perf_analyzer_path"] = self._perf_measurement_config.perf_analyzer_path

        return config

    def _get_perf_analyzer_flags(self):
        configuration = {}
        if self._profiling_data_path:
            if TRITON_MODEL_ANALYZER_VERSION >= LooseVersion("1.8.0"):
                configuration["input-data"] = [self._profiling_data_path.as_posix()]
            else:
                configuration["input-data"] = self._profiling_data_path.as_posix()
        elif self._dataset_profile_config and self._dataset_profile_config.max_shapes:

            shapes = get_shape_params(self._dataset_profile_config)

            if TRITON_MODEL_ANALYZER_VERSION >= LooseVersion("1.8.0"):
                configuration["shape"] = shapes
            else:
                configuration["shape"] = " ".join(shapes)

        configuration["measurement-interval"] = self._perf_measurement_config.perf_measurement_interval
        configuration["measurement-mode"] = self._perf_measurement_config.perf_measurement_mode
        configuration["measurement-request-count"] = self._perf_measurement_config.perf_measurement_request_count

        configuration["shared-memory"] = self._perf_measurement_config.perf_measurement_shared_memory
        configuration[
            "output-shared-memory-size"
        ] = self._perf_measurement_config.perf_measurement_output_shared_memory_size

        return configuration

    def _get_profile_config_for_model(self, model_dir_name):

        original_model_config_path = self._triton_config.model_repository / model_dir_name / "config.pbtxt"
        original_model_config = TritonModelConfigGenerator.parse_triton_config_pbtxt(original_model_config_path)

        model_config = {}

        if self._profile_config.config_search_instance_counts:
            mapping = {DeviceKind.GPU: "KIND_GPU", DeviceKind.CPU: "KIND_CPU"}
            model_config["instance_group"] = [
                {"kind": mapping[kind], "count": counts}
                for kind, counts in self._profile_config.config_search_instance_counts.items()
            ]

        if self._profile_config.config_search_max_batch_sizes:
            model_config["max_batch_size"] = self._profile_config.config_search_max_batch_sizes

        if self._profile_config.config_search_preferred_batch_sizes:
            model_config["dynamic_batching"] = {
                "preferred_batch_size": self._profile_config.config_search_preferred_batch_sizes
            }

        if self._profile_config.config_search_backend_parameters:
            original_backend_parameters = original_model_config.backend_parameters_config.triton_backend_parameters
            original_backend_parameters = {
                param_name: {"string_value": [param_value]}
                for param_name, param_value in original_backend_parameters.items()
            }
            model_config["parameters"] = {
                **original_backend_parameters,
                **{
                    param_name: {"string_value": list(map(str, param_values))}
                    for param_name, param_values in self._profile_config.config_search_backend_parameters.items()
                },
            }

        configuration = {}
        if model_config:
            configuration["model_config_parameters"] = model_config

        parameters = {}
        if self._profile_config.config_search_concurrency:
            parameters["concurrency"] = self._profile_config.config_search_concurrency

        if self._profile_config.config_search_batch_sizes:
            parameters["batch_sizes"] = self._profile_config.config_search_batch_sizes

        configuration["parameters"] = parameters

        engine_count_per_device = original_model_config.instances_config.engine_count_per_device
        if self._profile_config.config_search_max_instance_count and engine_count_per_device:
            if len(set(engine_count_per_device)) > 1:
                raise ModelNavigatorProfileException(
                    "Triton Model config instance group have more than 1 device kind. "
                    "Use manual profile to swipe over instance group count"
                )
            elif DeviceKind.CPU in engine_count_per_device:
                configuration["cpu_only"] = True

        return configuration
