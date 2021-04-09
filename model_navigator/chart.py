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
from typing import Dict, List, Tuple, Type, Union

import logging
import shutil
from pathlib import Path

import yaml

from .analysis import Analysis
from .catalog import Catalog
from .config import ModelNavigatorBaseConfig
from .framework import PyTorch, TensorFlow2
from .kubernetes import DeployerParameters, TesterParameters, generator
from .model import InputModel, Model
from .perf_analyzer import PerfAnalyzerConfig
from .result import Summary
from .tensor import TensorSpec

LOGGER = logging.getLogger(__name__)


class CustomDumper(yaml.Dumper):
    def ignore_aliases(self, data: Dict) -> bool:
        return True


_SUFFIX2FRAMEWORK = {
    ".savedmodel": TensorFlow2,
    ".plan": PyTorch,
    ".onnx": PyTorch,
    ".pt": PyTorch,
}


class ChartGenerator:
    def __init__(self, config: ModelNavigatorBaseConfig):
        self._config = config
        self._charts_directory = Path.cwd() / Path("workspace/charts/")
        self._workspace_path = "${SHARED_DIR}/workspace"

    def run(self, base_model: InputModel, summary: Summary, catalog: Catalog):
        self._prepare_charts_directory()

        for idx in range(self._config.top_n_configs):
            try:
                result = summary.get(idx)
            except IndexError:
                LOGGER.warning("Not enough results.")
                break

            analysis = Analysis.from_analyzer_result(result=result)

            model = catalog.get(model_name=analysis.model_name)

            deployment_parameters = self._deployment_paramters(
                base_model=base_model, model=model, analysis=analysis, config=self._config
            )
            tester_parameters = self._tester_parameters(base_model=base_model, model=model)

            framework = self._get_framework(base_model.path)

            model_name = self._normalize(deployment_parameters.model_name)
            output_path = self._charts_directory / f"{model_name}-{idx + 1}"

            generator(
                output_path=output_path,
                container_version=self._container_version(),
                framework=framework,
                deployer_parameters=deployment_parameters,
                deployer_cmds=self._deploy_commands(model=model, base_model=base_model),
                tester_parameters=tester_parameters,
                tester_cmds=self._tester_commands(),
                create_dockerfile=True
            )

            LOGGER.info(f"Created Helm Chart model on position {idx + 1} in {output_path}")

        LOGGER.info(f"Helm Charts generated {self._charts_directory}")

    def _tester_commands(self) -> List[str]:

        commands = list()

        perf_analyzer_flags = {
            "server-url": "${TRITON_SERVER_URL}",
            "model-name": "${MODEL_NAME}",
            "batch-sizes": "${BATCH_SIZE}",
            "triton-instances": "${TRITON_INSTANCES}",
            "input-data": "random",
            "result-path": "${SHARED_DIR}/triton_performance_offline.csv",
        }

        if self._config.value_ranges:
            perf_analyzer_flags["input-data"] = "${SHARED_DIR}/random_data.json"
            profiling_data_flags = {
                "shapes": self._shapes_to_cli(self._config.max_shapes),
                "value-ranges": self._value_ranges_to_cli(self._config.value_ranges),
                "iterations": 128,
                "output-path": "${SHARED_DIR}/random_data.json",
            }
            flags = self._flags_to_cli(profiling_data_flags)
            commands.append(
                rf"""python -m model_navigator.cli.create_profiling_data \
                {flags}
                """
            )
        elif self._config.max_shapes:
            shapes = [PerfAnalyzerConfig.shape_param_from_tensor_spec(spec) for spec in self._config.max_shapes]
            perf_analyzer_flags["input-shape"] = shapes

        flags = self._flags_to_cli(perf_analyzer_flags)
        offline_command = rf"""
        python -m model_navigator.cli.run_offline_performance_test_on_triton \
        {flags}
        """
        commands.append(offline_command)

        perf_analyzer_flags["number-of-model-instances"] = "${TRITON_GPU_ENGINE_COUNT}"
        flags = self._flags_to_cli(perf_analyzer_flags)
        online_command = rf"""
        python -m model_navigator.cli.run_online_performance_test_on_triton \
        {flags}
        """
        commands.append(online_command)

        return commands

    def _deploy_commands(self, model: Model, base_model: InputModel) -> List[str]:
        commands = list()
        base_model_path = f"${{SHARED_DIR}}/model{base_model.path.suffix}"
        optimized_model = base_model_path

        download_flags = {
            "file-url": "${MODEL_URI}",
            "to-path": base_model_path,
        }
        download_cli_args = self._flags_to_cli(download_flags)
        download_command = rf"""
        python -m model_navigator.cli.download_file \
            {download_cli_args}
        """
        commands.append(download_command)

        # FIXME: Converter should handle that case
        if self._model_should_convert(base_model, model):
            optimized_model = f"${{SHARED_DIR}}/optimized{model.path.suffix}"
            convert_flags = {
                "model-path": base_model_path,
                "model-name": "${MODEL_NAME}",
                "target-format": model.format.value,
                "target-precisions": model.precision.value,
                "output-path": optimized_model,
            }

            if model.onnx_opset:
                convert_flags["onnx-opsets"] = model.onnx_opset

            if self._config.max_workspace_size:
                convert_flags["max-workspace-size"] = self._config.max_workspace_size

            if self._config.min_shapes:
                convert_flags["min-shapes"] = self._shapes_to_cli(self._config.min_shapes)

            if self._config.max_shapes:
                convert_flags["max-shapes"] = self._shapes_to_cli(self._config.max_shapes)

            if self._config.opt_shapes:
                convert_flags["opt-shapes"] = self._shapes_to_cli(self._config.opt_shapes)

            if self._config.inputs:
                convert_flags["inputs"] = self._shapes_to_cli(self._config.inputs)

            if self._config.outputs:
                convert_flags["outputs"] = self._shapes_to_cli(self._config.outputs)

            if self._config.outputs:
                convert_flags["value-ranges"] = self._value_ranges_to_cli(self._config.value_ranges)

            if self._config.rtol:
                convert_flags["rtol"] = self._tolarence_to_cli(self._config.rtol)

            if self._config.atol:
                convert_flags["atol"] = self._tolarence_to_cli(self._config.atol)

            convert_cli_args = self._flags_to_cli(convert_flags)
            convert_command = rf"""
            python -m model_navigator.cli.convert_model \
                {convert_cli_args}
            """
            commands.append(convert_command)

        deploy_flags = {
            "model-repository": "${MODEL_REPOSITORY_PATH}",
            "model-path": optimized_model,
            "model-format": "${FORMAT}",
            "model-name": "${MODEL_NAME}",
            "model-version": "${MODEL_VERSION}",
            "max-batch-size": "${MAX_BATCH_SIZE}",
            "precision": "${PRECISION}",
            "number-of-model-instances": "${TRITON_GPU_ENGINE_COUNT}",
            "max-queue-delay-us": "${TRITON_MAX_QUEUE_DELAY}",
            "preferred-batch-sizes": "${TRITON_PREFERRED_BATCH_SIZES}",
            "capture-cuda-graph": "${CAPTURE_CUDA_GRAPH}",
            "backend-accelerator": "${ACCELERATOR}",
            "load-model": "none",
        }
        deploy_cli_args = self._flags_to_cli(deploy_flags)
        deploy_command = rf"""
                python -m model_navigator.cli.config_model_on_triton \
                    {deploy_cli_args}
                """
        commands.append(deploy_command)

        return commands

    def _deployment_paramters(
        self, base_model: InputModel, model: Model, analysis: Analysis, config: ModelNavigatorBaseConfig
    ) -> DeployerParameters:
        parameters = DeployerParameters(
            model_name=base_model.name,
            format=model.format.value,
            precision=model.precision.value,
            max_batch_size=max(analysis.preferred_batch_sizes),
            accelerator=model.accelerator.value,
            capture_cuda_graph=model.capture_cuda_graph,
            triton_gpu_engine_count=analysis.engine_count,
            triton_preferred_batch_sizes=analysis.preferred_batch_sizes,
            triton_max_queue_delay=config.max_latency_ms or 1,
            model_version=1,
        )
        return parameters

    def _tester_parameters(self, base_model: InputModel, model: Model):
        parameters = TesterParameters(
            model_name=base_model.name,
            batch_size=str(model.max_batch_size),
            triton_instances=1,
            triton_gpu_engine_count=model.gpu_engine_count,
            triton_server_url=f"http://{base_model.name}-inference:8000",
        )
        return parameters

    def _model_should_convert(self, base_model: InputModel, model: Model) -> bool:
        # Do not convert if the same model as input
        return base_model.path != model.path

    def _get_framework(self, path: Path) -> Type[Union[PyTorch, TensorFlow2]]:
        model_path = Path(path)
        suffix = model_path.suffix
        framework = _SUFFIX2FRAMEWORK[suffix]

        return framework

    def _prepare_charts_directory(self):
        if self._charts_directory.is_dir():
            shutil.rmtree(self._charts_directory.as_posix())

        self._charts_directory.mkdir(parents=True)

    def _normalize(self, str) -> str:
        normalized_name = str.lower().replace("_", "-")
        return f"{normalized_name}"

    def _container_version(self):
        chunks = self._config.triton_version.split("-")
        version = "-".join(chunks[: len(chunks) - 1])
        return version

    def _flags_to_cli(self, flags: Dict) -> str:
        flags_str = list()
        for key, value in flags.items():
            if not value:
                continue

            if isinstance(value, list):
                for item in value:
                    flags_str.append(f"--{key} {item}")
            else:
                flags_str.append(f"--{key} {value}")

        return " \\\n".join(flags_str)

    def _shapes_to_cli(self, shapes: List) -> str:
        cli_list = [TensorSpec.to_command_line(s) for s in shapes]
        return " ".join(cli_list)

    def _value_ranges_to_cli(self, value_ranges: List) -> str:
        values_cli = list()
        for value_range in value_ranges:
            name = value_range[0]
            min_value = value_range[1][0]
            max_value = value_range[1][1]
            values = ",".join(map(str, [min_value, max_value]))
            value_cli_item = ":".join([name, values])
            values_cli.append(value_cli_item)
        return " ".join(values_cli)

    def _tolarence_to_cli(self, tolerances: List[Tuple[str, float]]) -> str:
        return " ".join([f"{name}:{value}" if name else str(value) for name, value in tolerances])
