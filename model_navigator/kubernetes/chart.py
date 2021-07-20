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
import logging
from pathlib import Path
from typing import List, Optional, Type, Union

from model_navigator.cli.convert_model import ConversionSetConfig
from model_navigator.converter.config import (
    ComparatorConfig,
    ConversionConfig,
    ConversionLaunchMode,
    DatasetProfileConfig,
)
from model_navigator.framework import PyTorch, TensorFlow2
from model_navigator.kubernetes import internals
from model_navigator.kubernetes.generator import generator
from model_navigator.kubernetes.inference import InferenceChartCreator
from model_navigator.kubernetes.results import HelmChartGenerationResult
from model_navigator.model import ModelConfig, ModelSignatureConfig
from model_navigator.results import State, Status
from model_navigator.triton.config import (
    TritonModelInstancesConfig,
    TritonModelOptimizationConfig,
    TritonModelSchedulerConfig,
)
from model_navigator.utils.config import YamlConfigFile
from model_navigator.utils.formats import FORMAT2SUFFIX

LOGGER = logging.getLogger(__name__)

# TODO: refactor this
_SUFFIX2FRAMEWORK = {
    ".savedmodel": TensorFlow2,
    ".plan": PyTorch,
    ".onnx": PyTorch,
    ".pt": PyTorch,
}


class ChartGenerator:
    def __init__(self, *, container_version: str):
        """Generate Helm Chart"""
        self._docker_workspace_path = "${SHARED_DIR}/workspace"
        self._container_version = container_version

    def run(
        self,
        *,
        src_model: ModelConfig,
        src_model_signature_config: Optional[ModelSignatureConfig],
        conversion_config: ConversionConfig,
        comparator_config: ComparatorConfig,
        dataset_profile_config: DatasetProfileConfig,
        optimization_config: TritonModelOptimizationConfig,
        scheduler_config: TritonModelSchedulerConfig,
        instances_config: TritonModelInstancesConfig,
        output_path: Path,
    ):
        framework = self._get_framework(src_model.model_path)
        generator(
            chart_name=src_model.model_name,
            output_path=output_path,
            container_version=self._container_version,
            framework=framework,
            navigator_cmds=self._navigator_commands(src_model=src_model, conversion_config=conversion_config),
            evaluator_cmds=self._evaluator_commands(src_model=src_model, scheduler_config=scheduler_config),
            create_dockerfile=True,
        )

        # dump config file
        config_path = output_path / "config.yaml"
        conversion_set_config = ConversionSetConfig.from_single_config(conversion_config)
        with YamlConfigFile(config_path) as config_file:
            # do not save model_path from src_model
            # save just model_name and version
            config_file.save_key("model_name", src_model.model_name)
            config_file.save_key("model_version", src_model.model_version)
            config_file.save_config(src_model_signature_config)
            config_file.save_config(comparator_config)
            config_file.save_config(dataset_profile_config)
            config_file.save_config(optimization_config)
            config_file.save_config(scheduler_config)
            config_file.save_config(instances_config)

            if conversion_config.target_format:
                config_file.save_config(conversion_set_config)

        message = f"Created Helm Chart model for {src_model.model_name} in {output_path}"
        LOGGER.info(message)
        return HelmChartGenerationResult(
            status=Status(state=State.SUCCEEDED, message=message),
            container_version=self._container_version,
            src_model_config=src_model,
            src_model_signature_config=src_model_signature_config,
            conversion_config=conversion_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
            optimization_config=optimization_config,
            scheduler_config=scheduler_config,
            instances_config=instances_config,
            helm_chart_dir_path=output_path,
        )

    def _navigator_commands(self, src_model: ModelConfig, conversion_config: ConversionConfig) -> List[str]:
        model_suffix = src_model.model_path.suffix

        docker_src_model_path = f"${{SHARED_DIR}}/model{model_suffix}"
        commands = []

        commands.append(
            rf"""
            model-navigator download-model \
                --model-uri ${{MODEL_URI}} \
                --save-to {docker_src_model_path}
            """
        )

        converted_model_suffix = FORMAT2SUFFIX[conversion_config.target_format]
        docker_converted_model_path = f"${{SHARED_DIR}}/model_converted{converted_model_suffix}"
        commands.append(
            rf"""
            model-navigator convert \
                --config-path {internals.Paths.CONFIG_PATH} \
                --model-path {docker_src_model_path} \
                --output-path {docker_converted_model_path} \
                --launch-mode {ConversionLaunchMode.LOCAL.value}
            """,
        )

        commands.append(
            rf"""
            model-navigator triton-config-model \
                --config-path {internals.Paths.CONFIG_PATH} \
                --model-path {docker_converted_model_path} \
                --model-repository {internals.Paths.MODEL_REPOSITORY_PATH}
            """
        )

        return commands

    def _evaluator_commands(self, src_model: ModelConfig, scheduler_config: TritonModelSchedulerConfig) -> List[str]:
        server_url = f"http://{src_model.model_name.lower().replace('_', '-')}-{InferenceChartCreator.NAME}:8000"
        commands = [
            rf"""
            model-navigator triton-evaluate-model \
                --config-path {internals.Paths.CONFIG_PATH} \
                --server-url  {server_url} \
                --evaluation-mode static \
                --evaluation-mode dynamic \
                --max-batch-size {scheduler_config.max_batch_size} \
                --model-version {src_model.model_version} \
                --verbose
            """,
        ]
        return commands

    def _get_framework(self, path: Path) -> Type[Union[PyTorch, TensorFlow2]]:
        model_path = Path(path)
        suffix = model_path.suffix
        framework = _SUFFIX2FRAMEWORK[suffix]

        return framework

    def _normalize(self, name) -> str:
        return name.lower().replace("_", "-")
