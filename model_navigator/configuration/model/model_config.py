# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

"""This module contains classes representing model configurations."""

import pathlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from model_navigator.configuration import (
    Format,
    JitType,
    OnnxConfig,
    TensorRTCompatibilityLevel,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TensorRTProfile,
)
from model_navigator.configuration.runner.runner_config import DeviceRunnerConfig, TorchRunnerConfig
from model_navigator.utils.common import DataObject
from model_navigator.utils.format_helpers import FORMAT2SUFFIX, is_source_format


class ModelConfig(ABC, DataObject):
    """Abstract model configuration class."""

    _subclasses = {}
    format: Format

    def __init_subclass__(cls, format: Optional[Format], **kwargs):
        """Initializes ModelConfig subclass with format argument and adds it to dictionary.

        Args:
            format: Model format of initialized ModelConfig subclass
            kwargs: Additional subclass keyword arguments
        """
        super().__init_subclass__(**kwargs)
        if format:
            cls.format = format
            cls._subclasses[format] = cls

    def __new__(cls, *_, **__):
        """Returns new instance of class and sets format class variable.

        Ignores all other arguments.

        Returns:
            New instance of ModelConfig subclass
        """
        instance = super().__new__(cls)
        instance.format = cls.format
        return instance

    def __init__(self, parent: Optional["ModelConfig"], custom_args: Optional[Dict[str, Any]] = None) -> None:
        """Initializes ModelConfig class.

        Args:
            parent: Parent model configuration
            custom_args: Additional keyword arguments used for model export and conversions
        """
        self.parent = parent
        self.custom_args = custom_args or {}

    @classmethod
    def from_dict(cls, data_dict: Dict):
        """Creates ModelConfig from dictionary.

        Takes dictionary and use it to create appropriate ModelConfig subclass.
        data_dict['format'] is used to determine subclass.

        Args:
            data_dict: Dictionary with model configuration data

        Returns:
            Subclass of ModelConfig representing particular model configuration
        """
        return cls._subclasses[Format(data_dict["format"])]._from_dict(data_dict)

    def to_dict(self, *_, **__) -> dict:
        """Returns dictionary representation of the object.

        Instead of saving parent object, unique parent_path is saved.

        Ignores all other parameters.

        Returns:
            Dictionary representation of ModelConfig
        """
        params = {}
        for key, value in self.__dict__.items():
            if value is None:
                continue

            if hasattr(value, "to_dict") and not isinstance(value, ModelConfig):
                params = {**params, **value.to_dict()}
            else:
                params[key] = value

        return DataObject._from_dict({
            "format": self.format,
            "key": self.key,
            "path": self.path,
            "parent_path": self.parent_path,
            "parent_key": self.parent_key,
            "log_path": self.log_path,
            **params,
        })

    def get_config_dict_for_command(self) -> dict:
        """Returns dictionary with ModelConfig data required for Command execution.

        Returns:
            Dictionary representation of ModelConfig with unpacked params
        """
        return {
            "format": self.format,
            "key": self.key,
            "path": self.path,
            "log_path": self.log_path,
            "parent_key": self.parent_key,
            "parent_path": self.parent_path,
            **self.__dict__,
        }

    @property
    def key(self) -> str:
        """Get unique model key.

        Returns:
            str: model key.
        """
        config_hierarchy = []
        current_config = self
        while current_config is not None:
            config_hierarchy.append(current_config)
            current_config = current_config.parent

        key_params_array = [self.format.value]
        for c in config_hierarchy[::-1]:
            params = c._get_path_params_as_array_of_strings()
            if params:
                key_params_array.extend(params)

        key = "-".join(key_params_array)
        return key

    @property
    def path(self) -> pathlib.Path:
        """Get path to model checkpoint."""
        path_str = self.key

        if is_source_format(self.format):
            path = pathlib.Path(path_str) / "----"
        else:
            path = pathlib.Path(path_str) / f"model{FORMAT2SUFFIX[self.format]}"

        return path

    @property
    def log_path(self) -> pathlib.Path:
        """Get path to log file."""
        return self.path.parent / "format.log"

    @property
    def parent_path(self) -> Optional[pathlib.Path]:
        """Get path to parent model checkpoint."""
        if self.parent:
            return self.parent.path
        else:
            return None

    @property
    def parent_key(self) -> Optional[str]:
        """Get key of the parent model."""
        if self.parent:
            return self.parent.key
        else:
            return None

    @classmethod
    @abstractmethod
    def _from_dict(cls, data_dict: Dict):
        pass

    @staticmethod
    def _parse_string(parse_func: Callable, val: Optional[str] = None):
        """Parses string with parse_func or returns None if val not provided."""
        if val is not None:
            return parse_func(val)
        else:
            return None

    def _get_path_params_as_array_of_strings(self) -> List[str]:
        return []


class _SourceModelConfig(ModelConfig, format=None):
    """Source model configuration class."""

    def __init__(self) -> None:
        """Initializes base class for source code model configurations."""
        super().__init__(parent=None)


class PythonModelConfig(_SourceModelConfig, format=Format.PYTHON):
    """Source code Python model configuration class."""

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls()


class TorchModelConfig(_SourceModelConfig, format=Format.TORCH):
    """Source code Torch model configuration class."""

    def __init__(
        self,
        autocast: bool,
        inference_mode: bool,
        autocast_dtype: Optional[str] = None,
        device: Optional[str] = None,
        custom_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes Torch model configuration class.

        Args:
            autocast: Enable Automatic Mixed Precision in runner
            inference_mode: Enable inference mode in runner
            autocast_dtype: dtype used for autocast
            device: The target device on which mode has to be loaded
            custom_args: Additional keyword arguments used for model export and conversions
        """
        super().__init__()
        self.runner_config = TorchRunnerConfig(
            autocast=autocast,
            inference_mode=inference_mode,
            autocast_dtype=autocast_dtype,
            device=device,
            custom_args=custom_args,
        )

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls(
            autocast=cls._parse_string(bool, data_dict.get("autocast")),
            inference_mode=cls._parse_string(bool, data_dict.get("inference_mode")),
            device=data_dict.get("device"),
            autocast_dtype=data_dict.get("autocast_dtype"),
        )


class TensorFlowModelConfig(_SourceModelConfig, format=Format.TENSORFLOW):
    """Source code TensorFlow2 model configuration class."""

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls()


class JAXModelConfig(_SourceModelConfig, format=Format.JAX):
    """Source code JAX model configuration class."""

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls()


class _SerializedModelConfig(ModelConfig, format=None):
    """Serialized model configuration class."""

    pass


class TensorFlowSavedModelConfig(_SerializedModelConfig, format=Format.TF_SAVEDMODEL):
    """SavedModel model configuration class."""

    def __init__(
        self,
        jit_compile: Optional[bool],
        enable_xla: Optional[bool],
        parent: Optional[ModelConfig] = None,
        custom_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes SavedModel model configuration class.

        Args:
            parent: Parent model configuration
            jit_compile: Flag passed to tf.function in case of SavedModel exported from JAX model
            enable_xla: Flag passed to jax2tf.convert in case of SavedModel exported from JAX model
            custom_args: Custom arguments passed to Savedmodel export
        """
        super().__init__(parent=parent, custom_args=custom_args)
        self.jit_compile = jit_compile
        self.enable_xla = enable_xla

    def _get_path_params_as_array_of_strings(self) -> List[str]:
        params = []
        if self.jit_compile:
            params.append("jit")
        if self.enable_xla:
            params.append("xla")
        return params

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls(
            jit_compile=cls._parse_string(bool, data_dict.get("jit_compile")),
            enable_xla=cls._parse_string(bool, data_dict.get("enable_xla")),
        )


class TorchScriptModelConfig(_SerializedModelConfig, format=Format.TORCHSCRIPT):
    """TorchScript model configuration class."""

    def __init__(
        self,
        jit_type: JitType,
        strict: bool,
        autocast: bool,
        inference_mode: bool,
        autocast_dtype: Optional[str] = None,
        parent: Optional[ModelConfig] = None,
        custom_args: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes TorchScript model configuration class.

        Args:
            jit_type: TorchScript export method
            strict: Enable or Disable strict flag for tracer used in TorchScript export
            autocast: Enable Automatic Mixed Precision in runner
            inference_mode: Enable inference mode in runner
            autocast_dtype: dtype used for autocast
            parent: Parent model configuration
            custom_args: Custom arguments passed to TorchScript export
            device: runtime device e.g. "cuda:0"
        """
        super().__init__(parent=parent)
        self.jit_type = jit_type
        self.strict = strict
        self.custom_args = custom_args
        self.runner_config = TorchRunnerConfig(
            autocast=autocast,
            inference_mode=inference_mode,
            autocast_dtype=autocast_dtype,
            device=device,
        )

    def _get_path_params_as_array_of_strings(self) -> List[str]:
        return [self.jit_type.value] if self.jit_type else []

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls(
            jit_type=cls._parse_string(JitType, data_dict.get("jit_type")),
            strict=cls._parse_string(bool, data_dict.get("strict")),
            autocast=cls._parse_string(bool, data_dict.get("autocast")),
            inference_mode=cls._parse_string(bool, data_dict.get("inference_mode")),
            autocast_dtype=data_dict.get("autocast_dtype"),
            device=data_dict.get("device"),
        )


class TorchExportedProgramModelConfig(_SerializedModelConfig, format=Format.TORCH_EXPORTEDPROGRAM):
    """ExportedProgram model configuration class."""

    def __init__(
        self,
        autocast: bool,
        inference_mode: bool,
        parent: Optional[ModelConfig] = None,
        autocast_dtype: Optional[str] = None,
        custom_args: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes TorchScript model configuration class.

        Args:
            autocast: Enable Automatic Mixed Precision in runner
            inference_mode: Enable inference mode in runner
            parent: Parent model configuration
            autocast_dtype: dtype used for autocast
            custom_args: Custom arguments passed to TorchScript export
            device: runtime device e.g. "cuda:0"
        """
        super().__init__(parent=parent)
        self.custom_args = custom_args
        self.runner_config = TorchRunnerConfig(
            autocast=autocast, autocast_dtype=autocast_dtype, inference_mode=inference_mode, device=device
        )

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls(
            autocast=cls._parse_string(bool, data_dict.get("autocast")),
            inference_mode=cls._parse_string(bool, data_dict.get("inference_mode")),
            device=data_dict.get("device"),
            autocast_dtype=data_dict.get("autocast_dtype"),
        )


class ONNXExportEngine:
    """ONNX export engine base configuration class."""

    def __init__(self, engine_name: str) -> None:
        """Initialize ONNX export engine.

        Args:
            engine_name: Name of the export engine, for de/serialization purposes
        """
        self.engine_name = engine_name

    @staticmethod
    def _from_dict(data_dict: Optional[Dict] = None) -> Optional["ONNXExportEngine"]:
        """Create ONNX export engine from dictionary.

        Args:
            data_dict: Dictionary containing export engine configuration

        Returns:
            ONNX export engine
        """
        if data_dict is None:
            return None

        engine_name = data_dict.get("engine_name")
        if engine_name == OnnxDynamoExportConfig.NAME:
            return OnnxDynamoExportConfig(**data_dict)
        else:
            raise ValueError(f"Unknown export engine: {engine_name}")


class OnnxDynamoExportConfig(ONNXExportEngine):
    """Dynamo export engine."""

    NAME = "torch-dynamo"

    def __init__(self, dynamo_dynamic_shapes: Optional[bool] = None) -> None:
        """Initialize OnnxDynamoExportConfig.

        Args:
            dynamo_dynamic_shapes: Enable or Disable dynamic shapes in dynamo export
        """
        super().__init__(engine_name=OnnxDynamoExportConfig.NAME)
        self.dynamo_dynamic_shapes = dynamo_dynamic_shapes

    def to_dict(self) -> Dict:
        """Dict representation of ONNX export engine."""
        return {
            "engine_name": self.engine_name,
            "dynamo_dynamic_shapes": self.dynamo_dynamic_shapes,
        }


class ONNXModelConfig(_SerializedModelConfig, format=Format.ONNX):
    """ONNX model configuration class."""

    def __init__(
        self,
        opset: int,
        graph_surgeon_optimization: bool,
        parent: Optional[ModelConfig] = None,
        custom_args: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        export_device: Optional[str] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        model_path: Optional[Union[str, pathlib.Path]] = None,
        quantized: bool = False,
        export_engine: Optional[ONNXExportEngine] = None,
    ) -> None:
        """Initializes ONNX model configuration class.

        Args:
            opset: ONNX opset
            graph_surgeon_optimization: Enable or Disable Graph Surgeon optimization
            parent: Parent model configuration
            custom_args: Custom arguments passed to ONNX export
            device: runtime device e.g. "cuda:0"
            export_device: Device used for export
            dynamic_axes: Dynamic axes definition for ONNXConfig
            model_path: optional path to onnx model file, if provided the model will be loaded from the file instead of exporting to ONNX
            quantized: True if the model is quantized, default: False
            export_engine: Export engine config to use
        """
        super().__init__(parent=parent)
        self.opset = opset
        self.graph_surgeon_optimization = graph_surgeon_optimization
        self.custom_args = custom_args
        self.export_device = export_device
        # self.target_device = export_device
        self.runner_config = DeviceRunnerConfig(device=device)
        self.dynamic_axes = dynamic_axes
        self.model_path = model_path
        self.quantized = quantized
        self.export_engine = export_engine

    @staticmethod
    def from_onnx_config(
        onnx_config: OnnxConfig,
        parent: Optional[ModelConfig] = None,
        export_engine: Optional[ONNXExportEngine] = None,
    ) -> "ONNXModelConfig":
        """Create ONNX model configuration from ONNX config.

        Args:
            onnx_config: ONNX config
            parent: Parent model configuration
            export_engine: Export engine config to use
        """
        return ONNXModelConfig(
            parent=parent,
            opset=onnx_config.opset,
            graph_surgeon_optimization=onnx_config.graph_surgeon_optimization,
            custom_args=onnx_config.custom_args,
            device=onnx_config.device,
            export_device=onnx_config.export_device,
            dynamic_axes=onnx_config.dynamic_axes,
            export_engine=export_engine,
        )

    def _get_path_params_as_array_of_strings(self) -> List[str]:
        params_array = []
        if self.quantized:
            params_array.append("quantized")
        if self.export_engine is not None and self.export_engine.engine_name == OnnxDynamoExportConfig.NAME:
            params_array.append("dynamo")
        return params_array

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        return cls(
            opset=data_dict.get("opset"),
            graph_surgeon_optimization=data_dict.get("graph_surgeon_optimization"),
            custom_args=data_dict.get("custom_args"),
            device=data_dict.get("device"),
            export_device=data_dict.get("export_device"),
            model_path=data_dict.get("model_path"),
            quantized=data_dict.get("quantized", False),
            export_engine=ONNXExportEngine._from_dict(data_dict.get("export_engine")),
        )


class TensorRTModelConfig(_SerializedModelConfig, format=Format.TENSORRT):
    """TensorRT model configuration class."""

    def __init__(
        self,
        precision_mode: TensorRTPrecisionMode,
        optimization_level: Optional[int],
        compatibility_level: Optional[TensorRTCompatibilityLevel],
        max_workspace_size: Optional[int] = None,
        precision: Optional[TensorRTPrecision] = None,
        trt_profiles: Optional[List[TensorRTProfile]] = None,
        parent: Optional[ModelConfig] = None,
        onnx_parser_flags: Optional[List[int]] = None,
        custom_args: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        timing_cache_dir: Optional[str] = None,
        model_path: Optional[Union[str, pathlib.Path]] = None,
        conversion_fallback: bool = False,
    ) -> None:
        """Initializes TensorRT (plan) model configuration class.

        Args:
            parent: Parent model configuration/
            precision_mode: Mode how the precision flags are combined
            optimization_level: Level of TensorRT engine optimization
            compatibility_level: Hardware compatibility level
            max_workspace_size: The maximum GPU memory the model can use temporarily during execution
            precision: TensorRT model precision
            trt_profiles: TensorRT profiles
            onnx_parser_flags: ONNX parser flags
            custom_args: Custom arguments passed to TensorRT conversion
            device: runtime device e.g. "cuda:0"
            timing_cache_dir: Directory to store timing cache
            model_path: optional path to trt model file, if provided the model will be loaded from the file instead of converting ONNX to TRT
            conversion_fallback: Enable fallback for conversion to try conversion with smaller batch size
        """
        super().__init__(parent=parent)
        self.precision = precision
        self.precision_mode = precision_mode
        self.max_workspace_size = max_workspace_size
        self.trt_profiles = trt_profiles
        self.optimization_level = optimization_level
        self.compatibility_level = compatibility_level
        self.onnx_parser_flags = onnx_parser_flags
        self.custom_args = custom_args
        self.runner_config = DeviceRunnerConfig(device=device)
        self.timing_cache_dir = timing_cache_dir
        self.model_path = model_path
        self.conversion_fallback = conversion_fallback

    def _get_path_params_as_array_of_strings(self) -> List[str]:
        return [self.precision.value] if self.precision else []

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        trt_profiles = data_dict.get("trt_profiles")
        if trt_profiles is not None:
            trt_profiles = [TensorRTProfile.from_dict(trt_profile) for trt_profile in trt_profiles]
        onnx_parser_flags = data_dict.get("onnx_parser_flags")
        if onnx_parser_flags:
            onnx_parser_flags = [int(flag) for flag in onnx_parser_flags]
        return cls(
            precision=cls._parse_string(TensorRTPrecision, data_dict.get("precision")),
            precision_mode=cls._parse_string(TensorRTPrecisionMode, data_dict.get("precision_mode")),
            optimization_level=cls._parse_string(int, data_dict.get("optimization_level")),
            compatibility_level=cls._parse_string(TensorRTCompatibilityLevel, data_dict.get("compatibility_level")),
            max_workspace_size=cls._parse_string(int, data_dict.get("max_workspace_size")),
            trt_profiles=trt_profiles,
            onnx_parser_flags=onnx_parser_flags,
            timing_cache_dir=data_dict.get("timing_cache_dir"),
            model_path=data_dict.get("model_path"),
            conversion_fallback=data_dict.get("conversion_fallback", False),
        )


class TensorFlowTensorRTModelConfig(_SerializedModelConfig, format=Format.TF_TRT):
    """TensorFlow TensorRT model configuration class."""

    def __init__(
        self,
        precision: TensorRTPrecision,
        max_workspace_size: int,
        minimum_segment_size: int,
        trt_profiles: Optional[List[TensorRTProfile]] = None,
        parent: Optional[ModelConfig] = None,
        custom_args: Optional[Dict[str, Any]] = None,
        conversion_fallback: bool = False,
    ) -> None:
        """Initializes TensorFlow TensorRT model configuration class.

        Args:
            parent: Parent model configuration
            precision: TensorRT model precision
            max_workspace_size: TensorRT max workspace size
            minimum_segment_size: TensorRT minimum segment size
            trt_profiles: TensorRT profiles
            custom_args: Custom arguments passed to TensorFlow TensorRT conversion
            conversion_fallback: Enable fallback for conversion to try conversion with smaller batch size
        """
        super().__init__(parent=parent)
        self.precision = precision
        self.max_workspace_size = max_workspace_size
        self.minimum_segment_size = minimum_segment_size
        self.trt_profiles = trt_profiles
        self.custom_args = custom_args
        self.conversion_fallback = conversion_fallback

    def _get_path_params_as_array_of_strings(self) -> List[str]:
        return [self.precision.value] if self.precision else []

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        trt_profiles = data_dict.get("trt_profiles")
        if trt_profiles is not None:
            trt_profiles = [TensorRTProfile.from_dict(trt_profile) for trt_profile in trt_profiles]
        return cls(
            precision=cls._parse_string(TensorRTPrecision, data_dict.get("precision")),
            max_workspace_size=cls._parse_string(int, data_dict.get("max_workspace_size")),
            minimum_segment_size=cls._parse_string(int, data_dict.get("minimum_segment_size")),
            trt_profiles=trt_profiles,
            conversion_fallback=data_dict.get("conversion_fallback", False),
        )


class TorchTensorRTModelConfig(_SerializedModelConfig, format=Format.TORCH_TRT):
    """Torch TensorRT model configuration class."""

    def __init__(
        self,
        precision: TensorRTPrecision,
        precision_mode: TensorRTPrecisionMode,
        max_workspace_size: int,
        pickle_protocol: int,
        trt_profiles: Optional[List[TensorRTProfile]] = None,
        parent: Optional[ModelConfig] = None,
        custom_args: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        conversion_fallback: bool = False,
    ) -> None:
        """Initializes Torch TensorRT model configuration class.

        Args:
            parent: Parent model configuration
            precision: TensorRT model precision
            precision_mode: Mode how the precision flags are combined
            max_workspace_size: The maximum GPU memory the model can use temporarily during execution
            pickle_protocol: Pickle protocol used during model serialization
            trt_profiles: TensorRT profiles
            custom_args: Custom arguments passed to Torch TensorRT conversion
            device: runtime device e.g. "cuda:0"
            conversion_fallback: Enable fallback for conversion to try conversion with smaller batch size
        """
        super().__init__(parent=parent)
        self.precision = precision
        self.precision_mode = precision_mode
        self.max_workspace_size = max_workspace_size
        self.pickle_protocol = pickle_protocol
        self.trt_profiles = trt_profiles
        self.custom_args = custom_args
        self.runner_config = DeviceRunnerConfig(device=device)
        self.conversion_fallback = conversion_fallback

    def _get_path_params_as_array_of_strings(self) -> List[str]:
        return [self.precision.value] if self.precision else []

    @classmethod
    def _from_dict(cls, data_dict: Dict):
        trt_profiles = data_dict.get("trt_profiles")
        if trt_profiles is not None:
            trt_profiles = [TensorRTProfile.from_dict(trt_profile) for trt_profile in trt_profiles]
        return cls(
            precision=cls._parse_string(TensorRTPrecision, data_dict.get("precision")),
            precision_mode=cls._parse_string(TensorRTPrecisionMode, data_dict.get("precision_mode")),
            max_workspace_size=cls._parse_string(int, data_dict.get("max_workspace_size")),
            pickle_protocol=cls._parse_string(int, data_dict.get("pickle_protocol")),
            trt_profiles=trt_profiles,
            device=data_dict.get("device"),
            conversion_fallback=data_dict.get("conversion_fallback", False),
        )
