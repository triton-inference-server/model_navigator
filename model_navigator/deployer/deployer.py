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
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from tritonclient.utils import InferenceServerException

from . import utils
from ..config import ModelNavigatorBaseConfig
from ..device import utils as device_utils
from ..log import FileLogger, section_header
from ..model import Model
from ..model_navigator_exceptions import ModelNavigatorDeployerException, ModelNavigatorException
from ..perf_analyzer import PerfAnalyzer, PerfAnalyzerConfig
from ..perf_analyzer.profiling_data import get_profiling_data_path
from ..tensor import TensorSpec
from ..triton import (
    ModelConfig,
    TritonClient,
    TritonModelStore,
    TritonServer,
    TritonServerConfig,
    TritonServerException,
    TritonServerFactory,
)

LOGGER = logging.getLogger(__name__)


class Deployer:
    def __init__(self, config: ModelNavigatorBaseConfig):
        self._config = config
        workspace_dir = Path(config.workspace_path)
        self._model_repository = workspace_dir / "model-store"
        self._profiling_data_path = get_profiling_data_path(workspace_dir)

    def deploy_model(self, model: Model):
        self._prepare_model_repository()

        model_config = ModelConfig.create(
            model_path=model.path,
            model_name=model.name,
            model_version="1",
            model_format=model.format.value,
            max_batch_size=model.max_batch_size,
            precision=model.precision.value,
            gpu_engine_count=model.gpu_engine_count,
            preferred_batch_sizes=self._get_preferred_batch_sizes(max_batch_size=model.max_batch_size),
            max_queue_delay_us=1,
            capture_cuda_graph=model.capture_cuda_graph,
            accelerator=model.accelerator.value,
        )
        server_ip = "localhost"
        server_port = "8001"

        server = self._get_server()
        try:
            server.start()

            server_url = f"grpc://{server_ip}:{server_port}"
            client = TritonClient(server_url=server_url)
            client.wait_for_server_ready(timeout=60)

            model_store = TritonModelStore(self._model_repository)
            path_in_model_store = model_store.deploy_model(model_config=model_config, model_path=model.path)

            client.load_model(model_name=model_config.model_name)

            model_metadata = client.wait_for_model(
                model_name=model_config.model_name, model_version=model_config.model_version
            )

            server_url = f"{server_ip}:{server_port}"
            params = {
                "model-name": model_config.model_name,
                "batch-size": model_config.max_batch_size,
                "model-version": 1,
                "url": server_url,
                "protocol": "grpc",
                "measurement-interval": 10000,
            }

            shape_params = self._get_shape_params(model_metadata)
            if self._config.value_ranges:
                params["input-data"] = self._profiling_data_path.as_posix()
            elif shape_params:
                params["shape"] = shape_params

            perf_config = PerfAnalyzerConfig()
            for param, value in params.items():
                perf_config[param] = value

            perf_analyzer = PerfAnalyzer(perf_config)

            LOGGER.info(f"Evaluating model {model_config.model_name} on Triton")
            perf_analyzer.run()
            LOGGER.debug(perf_analyzer.output())
        except (ModelNavigatorException, TritonServerException, InferenceServerException) as e:
            server.stop()

            error = e.message()
            LOGGER.debug(error)

            logs = server.logs()
            LOGGER.debug(logs)

            log_file = self._log_error(model, logs, error)
            model.error_log = log_file

            raise ModelNavigatorDeployerException(e.message)
        finally:
            server.stop()

        LOGGER.info(f"Done. Model {model_config.model_name} ready to promote to analysis.")
        model.triton_path = path_in_model_store

    def _get_shape_params(self, model_metadata: Dict[str, Any]) -> Optional[str]:
        shape_params = None
        triton_input_specs = [TensorSpec.from_triton_tensor_metadata(metadata) for metadata in model_metadata["inputs"]]
        inputs_with_dynamic_axes = [spec for spec in triton_input_specs if spec.is_dynamic()]
        LOGGER.debug(
            f"Model {model_metadata['name']} inputs with dynamic axes: "
            f"{', '.join([spec.name for spec in inputs_with_dynamic_axes]) or '<no inputs with dynamic axes>'}"
        )

        missing_inputs = sorted(
            {spec.name for spec in inputs_with_dynamic_axes} - {spec.name for spec in self._config.max_shapes}
        )
        if missing_inputs:
            raise ModelNavigatorException(
                f"Couldn't find shape specification for performance analysis for inputs with dynamic axes: "
                f"{', '.join(missing_inputs)}. Use --max-shapes to define them."
            )
        elif inputs_with_dynamic_axes:
            shape_params = " ".join(
                [PerfAnalyzerConfig.shape_param_from_tensor_spec(spec) for spec in self._config.max_shapes]
            )
        return shape_params

    def _prepare_model_repository(self):
        if self._model_repository.is_dir():
            shutil.rmtree(self._model_repository.as_posix())

        self._model_repository.mkdir(parents=True)

    def _get_preferred_batch_sizes(self, max_batch_size: int) -> List[int]:
        batch_sizes = [max_batch_size // 2, max_batch_size]

        if any([batch_size > max_batch_size for batch_size in batch_sizes]):
            raise ValueError("Preferred batch size cannot be greater then max_batch_size.")

        triton_preferred_batch_sizes = list(map(int, filter(lambda x: x > 0, batch_sizes)))

        return triton_preferred_batch_sizes

    def _get_server(self) -> TritonServer:
        """
        Creates and returns a TritonServer
        with specified arguments

        Parameters
        ----------
        model_repository:
            Path to model repository
        config : namespace
            Arguments parsed from the CLI
        """
        triton_config = TritonServerConfig()
        triton_config["model-repository"] = self._model_repository.resolve().as_posix()
        triton_config["model-control-mode"] = "explicit"
        triton_config["strict-model-config"] = "false"

        if self._config.triton_launch_mode == "local":
            server = TritonServerFactory.create_server_local(path=self._config.triton_server_path, config=triton_config)
        elif self._config.triton_launch_mode == "docker":
            server = TritonServerFactory.create_server_docker(
                image="nvcr.io/nvidia/tritonserver:" + self._config.triton_version,
                config=triton_config,
                gpus=device_utils.get_gpus(config=self._config),
            )
        else:
            raise ModelNavigatorException(f"Unrecognized triton-launch-mode : {self._config.triton_launch_mode}")

        return server

    def _log_error(self, model: Model, server_log: str, error: str) -> Path:

        logger = FileLogger(name=model.name, config=self._config)
        header = utils.prepare_log_header(model)
        logger.log(header)

        logger.log(section_header("Client Error Log"))
        logger.log(error)

        logger.log(section_header("Triton Inference Server Log"))
        logger.log(server_log)

        return logger.file_path
