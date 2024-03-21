# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Collection of status classes.

Those classes are used to store information about the Model Navigator process.
"""

import collections
import dataclasses
import datetime
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from packaging import version

from model_navigator.api.config import (
    Format,
    OnnxConfig,
    OptimizationProfile,
    TensorFlowTensorRTConfig,
    TensorRTConfig,
    TensorRTPrecisionMode,
    TensorRTProfile,
    TorchTensorRTConfig,
)
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.correctness.correctness import Correctness, TolerancePerOutputName
from model_navigator.commands.performance.performance import Performance, ProfilingResults
from model_navigator.commands.verification.verify import VerifyModel
from model_navigator.configuration.model.model_config import ModelConfig
from model_navigator.core.constants import NAVIGATOR_PACKAGE_VERSION, NAVIGATOR_VERSION
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.frameworks import Framework
from model_navigator.frameworks.tensorrt.utils import get_trt_profile_from_trt_dynamic_axes
from model_navigator.utils.common import DataObject


@dataclasses.dataclass
class RunnerStatus(DataObject):
    """Runner results."""

    runner_name: str
    status: Dict[str, CommandStatus] = dataclasses.field(default_factory=lambda: {})
    result: Dict = dataclasses.field(default_factory=lambda: {})

    @classmethod
    def from_dict(cls, data_dict: Dict) -> "RunnerStatus":
        """Create RunnerStatus from the dictionary.

        Args:
            data_dict (Dict): dictionary with runner results data.

        Returns:
            RunnerStatus
        """
        status = {k: CommandStatus(v) for k, v in data_dict["status"].items()}
        result = {}

        for c, s in status.items():
            if s != CommandStatus.OK:
                continue

            if c == Correctness.name:
                result[c] = {
                    "per_output_tolerance": TolerancePerOutputName.from_json(
                        data_dict["result"][Correctness.__name__]["per_output_tolerance"]
                    )
                }
            elif c == Performance.name:
                result[c] = {
                    "profiling_results": [
                        ProfilingResults.from_dict(profiling_results_dict)
                        for profiling_results_dict in data_dict["result"][Performance.__name__]["profiling_results"]
                    ]
                }

        return cls(
            runner_name=data_dict["runner_name"],
            status=status,
            result=result,
        )


@dataclasses.dataclass
class ModelStatus(DataObject):
    """Model Status."""

    model_config: ModelConfig
    runners_status: Dict[str, RunnerStatus] = dataclasses.field(default_factory=lambda: {})
    status: Dict[str, CommandStatus] = dataclasses.field(default_factory=lambda: {})
    result: Dict = dataclasses.field(default_factory=lambda: {})

    @classmethod
    def from_dict(cls, data_dict: Dict) -> "ModelStatus":
        """Create ModelStatus from the dictionary.

        Args:
            data_dict (Dict): dictionary with model status data.

        Returns:
            ModelStatus
        """
        return cls(
            model_config=ModelConfig.from_dict(data_dict["model_config"]),
            runners_status={
                runner_name: RunnerStatus.from_dict(runner_res)
                for runner_name, runner_res in data_dict["runners_status"].items()
            },
            status={k: CommandStatus(v) for k, v in data_dict.get("status", {}).items()},
            result=data_dict.get("result", {}),
        )


@dataclasses.dataclass
class Status(DataObject):
    """Model Navigator Status."""

    format_version: str
    model_navigator_version: str
    uuid: str
    environment: Dict
    config: Dict
    models_status: Dict[str, ModelStatus]
    input_metadata: TensorMetadata
    output_metadata: TensorMetadata
    dataloader_trt_profile: TensorRTProfile
    dataloader_max_batch_size: int
    status: Dict[str, CommandStatus] = dataclasses.field(default_factory=lambda: {})
    result: Dict = dataclasses.field(default_factory=lambda: {})
    timestamp: str = dataclasses.field(default_factory=lambda: f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S.%f}")

    @classmethod
    def from_dict(cls, data_dict: Dict) -> "Status":
        """Create NavigatorStatus from the dictionary.

        Args:
            data_dict (Dict): Dictionary with navigator status data.

        Returns:
            NavigatorStatus
        """

        def _extract_format_version(status_dict):
            format_version = status_dict.get("format_version", "0.1.0")
            if format_version == "0.1.0" and "optimization_profile" in status_dict["export_config"]:
                format_version = "0.1.1"
            return version.parse(format_version)

        format_version = _extract_format_version(data_dict)
        StatusDictUpdater().update(data_dict, format_version)

        if isinstance(data_dict["config"].get("_input_names"), Sequence):
            data_dict["config"]["_input_names"] = tuple(data_dict["config"]["_input_names"])
        if isinstance(data_dict["config"].get("_output_names"), Sequence):
            data_dict["config"]["_output_names"] = tuple(data_dict["config"]["_output_names"])

        dataloader_trt_profile = TensorRTProfile()
        for name, val in data_dict["dataloader_trt_profile"].items():
            dataloader_trt_profile.add(name, **val)

        models_status = {
            model_key: ModelStatus.from_dict(models_status)
            for model_key, models_status in data_dict["models_status"].items()
        }
        # update model_config parents
        for model_status, model_status_dict in zip(models_status.values(), data_dict["models_status"].values()):
            for parent_model_key in models_status:
                if parent_model_key == model_status_dict["model_config"]["parent_key"]:
                    model_status.model_config.parent = models_status[parent_model_key].model_config

        return cls(
            format_version=NAVIGATOR_PACKAGE_VERSION,
            model_navigator_version=NAVIGATOR_VERSION,
            uuid=data_dict["uuid"],
            environment=data_dict["environment"],
            config=data_dict["config"],
            models_status=models_status,
            input_metadata=TensorMetadata.from_json(data_dict["input_metadata"]),
            output_metadata=TensorMetadata.from_json(data_dict["output_metadata"]),
            dataloader_trt_profile=dataloader_trt_profile,
            dataloader_max_batch_size=int(data_dict["dataloader_max_batch_size"]),
            timestamp=data_dict["timestamp"],
        )

    def get_model_configs(self) -> Dict[Format, List[ModelConfig]]:
        """Get model configurations from the status.

        Returns:
            Dict[Format, List[ModelConfig]]: Dictionary where key is a model format
                and value is a list of model configs for this format.
        """
        model_configs = collections.defaultdict(list)
        for models_status in self.models_status.values():
            model_configs[models_status.model_config.format].append(models_status.model_config)
        return model_configs


class StatusDictUpdater:
    """Update status dictionary to the current version."""

    def __init__(self):
        """Construct StatusDictUpdater."""
        self._updates = {
            version.parse("0.1.0"): self._update_from_v0_1_0,
            version.parse("0.1.2"): self._update_from_v0_1_2,
            version.parse("0.1.3"): self._update_from_v0_1_3,
            version.parse("0.1.4"): self._update_from_v0_1_4,
            version.parse("0.2.1"): self._update_from_v0_2_1,
            version.parse("0.2.2"): self._update_from_v0_2_2,
            version.parse("0.2.3"): self._update_from_v0_2_3,
            version.parse("0.3.0"): self._update_from_v0_3_0,
        }

    def update(self, data_dict: Dict, format_version: version.Version):
        """Update the status data dict from the format_version to the current version.

        Args:
            data_dict (Dict): Data dict to be updated.
            format_version (version.Version): Version of the data dict to be updated.
        """
        if format_version < version.parse("0.1.4"):
            LOGGER.warning(
                f"The version of this package is outdated. Your package's version is {format_version}, "
                f"but the current API version {NAVIGATOR_PACKAGE_VERSION} requires at least version 0.1.4. "
                "We do not guarantee that all Package API functions works correctly. "
                "Please, update your package running 'nav.package.optimize(package)' function."
            )
        for update_from_version, update_func in self._updates.items():
            if format_version <= update_from_version:
                update_func(data_dict)

    def _update_from_v0_1_0(self, data_dict: Dict):
        for i, model_status in enumerate(data_dict["model_status"]):
            for runtime_results in model_status["runtime_results"]:
                for i in range(len(runtime_results.get("performance", []))):
                    perf_results = runtime_results["performance"][i]
                    runtime_results["performance"][i] = {
                        "batch_size": perf_results["batch_size"],
                        "avg_latency": perf_results["latency"],
                        "std_latency": None,
                        "p50_latency": None,
                        "p90_latency": None,
                        "p95_latency": None,
                        "p99_latency": None,
                        "throughput": perf_results["throughput"],
                        "avg_gpu_clock": None,
                        "request_count": None,
                    }

        if data_dict["export_config"]["framework"] == "pytorch" and "precision_mode" not in data_dict["config"]:
            default_val = TensorRTPrecisionMode.SINGLE.value
            LOGGER.info(f"Using default `precision_mode`: {default_val}")
            data_dict["export_config"]["precision_mode"] = default_val
        if "optimization_profile" not in data_dict["export_config"]:
            default_val = OptimizationProfile().to_dict()
            LOGGER.info(f"Using default `optimization_profile`: {default_val}")
            data_dict["export_config"]["optimization_profile"] = default_val
        if (
            "git_info" not in data_dict
        ):  # FIXME probably git_info should be removed from package updater - git_info no longer saved in status.yaml
            data_dict["git_info"] = {}

    def _update_from_v0_1_2(self, data_dict: Dict[str, Any]):
        data_dict["trt_profile"] = DataObject.parse_value(
            get_trt_profile_from_trt_dynamic_axes(data_dict["export_config"]["trt_dynamic_axes"])
        )
        for model_status in data_dict["model_status"]:
            assert isinstance(model_status, dict)
            if model_status["format"] == "torch-trt" and model_status.get("precision") is None:
                model_status["precision"] = "fp32"

    def _update_from_v0_1_3(self, data_dict: Dict):
        if (
            data_dict["export_config"]["framework"] == "pytorch"
            and data_dict["export_config"].get("precision_mode") is None
        ):
            default_val = TensorRTPrecisionMode.SINGLE.value
            LOGGER.info(f"Using default `precision_mode`: {default_val}")
            data_dict["export_config"]["precision_mode"] = default_val
        if "onnx_runtimes" in data_dict["export_config"]:
            data_dict["export_config"]["runtimes"] = data_dict["export_config"].pop("onnx_runtimes")

    def _update_from_v0_1_4(self, data_dict: Dict):
        models_status = {}
        dataloader_max_batch_size = 1
        for i, model_status in enumerate(data_dict["model_status"]):
            model_path = model_status["path"]
            if not model_path:
                continue

            model_key = pathlib.Path(model_path).parent.name
            model_config_dict = {
                "key": model_key,
                "path": model_status.get("path"),
                "format": model_status.get("format"),
                "parent_path": None,
                "parent_key": None,
                "log_path": None,
            }
            if "precision" in model_status:
                model_config_dict["precision"] = model_status.get("precision")
            if "torch_jit" in model_status:
                model_config_dict["jit_type"] = model_status.get("torch_jit")
            if "enable_xla" in model_status:
                model_config_dict["enable_xla"] = model_status.get("enable_xla")
            if "jit_compile" in model_status:
                model_config_dict["jit_compile"] = model_status.get("jit_compile")

            data_dict["model_status"][i]["model_config"] = model_config_dict

            model_format = data_dict["model_status"][i]["format"]
            runtime_results = data_dict["model_status"][i]["runtime_results"]
            dataloader_max_batch_size, runners_status = self._prepare_runners_status_for_0_1_4(
                dataloader_max_batch_size,
                model_format,
                runtime_results,
            )

            data_dict["model_status"][i].pop("runtime_results")
            data_dict["model_status"][i]["runners_status"] = runners_status

            models_status[model_key] = data_dict["model_status"][i]

        data_dict.pop("model_status")
        data_dict["models_status"] = models_status

        export_config = data_dict.pop("export_config")
        config = self._config_from_0_1_4(export_config)
        data_dict["config"] = config
        data_dict["dataloader_max_batch_size"] = dataloader_max_batch_size
        data_dict["dataloader_trt_profile"] = data_dict.get("trt_profile") or {}

        data_dict["timestamp"] = export_config["timestamp"]

    def _framework2framework_for_0_1_4(self, framework: str) -> str:
        mapping = {
            "pytorch": Framework.TORCH.value,
            "tensorflow2": Framework.TENSORFLOW.value,
        }
        return mapping.get(framework, framework)

    def _runtime2runners_for_0_1_4(self, model_format: str, runtime_result: Dict) -> Optional[RunnerStatus]:
        format_and_runtime2runner = {
            Format.ONNX.value: {
                "CPUExecutionProvider": "OnnxCPU",
                "CUDAExecutionProvider": "OnnxCUDA",
                "TensorrtExecutionProvider": "OnnxTensorRT",
            },
            Format.TENSORRT.value: {
                "TensorrtExecutionProvider": "TensorRT",
                "TrtexecExecutionProvider": None,
            },
            Format.TF_TRT.value: {
                "TensorFlowExecutionProvider": "TensorFlowTensorRT",
            },
            Format.TF_SAVEDMODEL.value: {
                "TensorFlowExecutionProvider": "TensorFlowSavedModel",
            },
            Format.TORCH_TRT.value: {
                "PyTorchExecutionProvider": "TorchTensorRT",
            },
            Format.TORCHSCRIPT.value: {
                "PyTorchExecutionProvider": "TorchScriptCUDA",
            },
        }

        runner_name = format_and_runtime2runner[model_format][runtime_result["runtime"]]
        if not runner_name:
            return None

        tolerance = runtime_result["tolerance"]
        performance = runtime_result["performance"]
        verified = runtime_result["verified"]

        tolerance_status = CommandStatus.OK
        performance_status = CommandStatus.OK
        verify_status = CommandStatus.OK

        if not tolerance:
            tolerance_status = CommandStatus.FAIL
            performance_status = CommandStatus.SKIPPED
            verify_status = CommandStatus.SKIPPED
        elif not performance:
            performance_status = CommandStatus.FAIL
            verify_status = CommandStatus.SKIPPED
        elif not verified:
            verify_status = CommandStatus.SKIPPED

        result = {}
        if tolerance:
            result[Correctness.__name__] = {
                "per_output_tolerance": tolerance,
            }

        if performance:
            result[Performance.__name__] = {
                "profiling_results": performance,
            }

        runner_status = RunnerStatus(
            runner_name=runner_name,
            status={
                Correctness.__name__: tolerance_status,
                Performance.__name__: performance_status,
                VerifyModel.__name__: verify_status,
            },
            result=result,
        )

        return runner_status

    def _prepare_runners_status_for_0_1_4(self, dataloader_max_batch_size, model_format, runtime_results) -> Tuple:
        runners_status = {}
        for runtime_result in runtime_results:
            runner_status = self._runtime2runners_for_0_1_4(model_format, runtime_result)
            if not runner_status:
                continue

            dataloader_max_batch_size = self._get_dataloader_max_batch_size_for_0_1_4(
                dataloader_max_batch_size,
                runner_status=runner_status,
            )

            runners_status[runner_status.runner_name] = runner_status.to_dict()

        return dataloader_max_batch_size, runners_status

    def _get_dataloader_max_batch_size_for_0_1_4(
        self, dataloader_max_batch_size: int, runner_status: RunnerStatus
    ) -> int:
        performance_result = runner_status.result.get(Performance.__name__)
        if not performance_result:
            return dataloader_max_batch_size

        for profiling_result in performance_result.get("profiling_results", []):
            batch_size = int(profiling_result["batch_size"])
            if batch_size > dataloader_max_batch_size:
                dataloader_max_batch_size = batch_size

        return dataloader_max_batch_size

    def _config_from_0_1_4(self, export_config: Dict):
        framework = self._framework2framework_for_0_1_4(export_config["framework"])
        config = {
            "framework": framework,
        }

        valid_fields = [
            "target_formats",
            "sample_count",
            "batch_dim",
            "_input_names",
            "_output_names",
            "optimization_profile",
            "forward_kw_names",
            "target_device",
            "dynamic_axes",
            "opset",
            "trt_dynamic_axes",
        ]

        framework_based_classes = []
        if framework == Framework.TORCH:
            framework_based_classes = [TorchTensorRTConfig]
        elif framework in [Framework.TENSORFLOW, Framework.JAX]:
            framework_based_classes = [TensorFlowTensorRTConfig]

        custom_configs_fields = {
            "dynamic_axes": [OnnxConfig],
            "opset": [OnnxConfig],
            "trt_profile": [TensorRTConfig] + framework_based_classes,
        }

        custom_configs = {}
        for field_name, value in export_config.items():
            if field_name not in valid_fields:
                continue
            if field_name == "trt_dynamic_axes":
                field_name = "trt_profile"
                value = get_trt_profile_from_trt_dynamic_axes(value)

            custom_config_classes = custom_configs_fields.get(field_name)
            if custom_config_classes:
                for config_class in custom_config_classes:
                    obj = custom_configs.get(config_class.name(), {})
                    obj[field_name] = value
                    custom_configs[config_class.name()] = obj
            else:
                config[field_name] = value

        config["custom_configs"] = custom_configs

        if "runtimes" in export_config:
            config["runner_names"] = export_config["runtimes"]

        return config

    def _update_from_v0_2_1(self, data_dict: Dict):
        config = data_dict["config"]
        optimization_profile = {}
        if "profiler_config" in config:
            profiler_config = config.pop("profiler_config")

            if "run_profiling" in profiler_config:
                profiler_config.pop("run_profiling")

            optimization_profile = {
                "batch_sizes": profiler_config["batch_sizes"],
                "window_size": profiler_config["measurement_request_count"],
                "stability_percentage": profiler_config["stability_percentage"],
                "max_trials": profiler_config["max_trials"],
                "throughput_cutoff_threshold": profiler_config["throughput_cutoff_threshold"],
            }

        if optimization_profile:
            config["optimization_profile"] = optimization_profile

    def _update_from_v0_2_2(self, data_dict: Dict):
        config = data_dict["config"]
        for custom_config in config["custom_configs"].values():
            if "trt_profile" in custom_config:
                trt_profile = custom_config["trt_profile"]
                if trt_profile is not None:
                    custom_config["trt_profiles"] = [trt_profile]
                custom_config.pop("trt_profile")

    def _update_from_v0_2_3(self, data_dict: Dict):
        def _update_tensor_metadata(tensor_metadata: Dict, forward_kw_names: Optional[List[str]] = None):
            if forward_kw_names is None:
                pytree_metadata = tuple(data["name"] for data in tensor_metadata)
                if len(pytree_metadata) == 1:
                    pytree_metadata = pytree_metadata[0]
            else:
                pytree_metadata = {
                    forward_kw_name: data["name"] for forward_kw_name, data in zip(forward_kw_names, tensor_metadata)
                }
            updated_metadata = {
                "metadata": tensor_metadata,
                "pytree_metadata": {
                    "metadata": pytree_metadata,
                    "tensor_type": "numpy",
                },
            }
            return updated_metadata

        data_dict["input_metadata"] = _update_tensor_metadata(
            data_dict["input_metadata"], data_dict["config"].get("forward_kw_names")
        )
        data_dict["output_metadata"] = _update_tensor_metadata(data_dict["output_metadata"])
        data_dict["output_metadata"]["is_legacy"] = True

        if "forward_kw_names" in data_dict["config"]:
            data_dict["config"].pop("forward_kw_names")

    def _update_from_v0_3_0(self, data_dict: Dict):
        config = data_dict["config"]
        custom_configs = config.pop("custom_configs", {})
        if "Torch" in custom_configs:
            torch_config = custom_configs.pop("Torch")
            custom_configs["TorchScript"] = torch_config

        config["custom_configs"] = custom_configs
