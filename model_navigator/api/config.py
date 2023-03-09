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
"""Definition of enums and classes representing input configuration for Model Navigator."""
import abc
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import numpy

from model_navigator.constants import (
    DEFAULT_MAX_WORKSPACE_SIZE,
    DEFAULT_MIN_SEGMENT_SIZE,
    DEFAULT_ONNX_OPSET,
    DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD,
)
from model_navigator.logger import LOGGER
from model_navigator.utils.common import DataObject
from model_navigator.utils.framework import Framework

Sample = Dict[str, numpy.ndarray]

VerifyFunction = Callable[[Iterable[Sample], Iterable[Sample]], bool]


class DeviceKind(Enum):
    """Support types of devices in runners."""

    CPU = "cpu"
    CUDA = "cuda"


@runtime_checkable
class SizedIterable(Protocol):
    """Protocol representing sized iterable. Used by dataloader."""

    def __iter__(self) -> Iterator:
        """Magic method __iter__.

        Returns:
            Iterator to next item.
        """
        ...

    def __len__(self) -> int:
        """Magic method __len__.

        Returns:
            Length of size iterable.
        """
        ...


SizedDataLoader = Union[SizedIterable, Sequence]


class Format(Enum):
    """All model formats supported by Model Navigator 'optimize' function."""

    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    TORCHSCRIPT = "torchscript"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TORCH_TRT = "torch-trt"
    ONNX = "onnx"
    TENSORRT = "trt"


class JitType(Enum):
    """TorchScript export paramter."""

    SCRIPT = "script"
    TRACE = "trace"


class TensorRTPrecision(Enum):
    """Precisions supported during TensorRT conversions."""

    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


class TensorRTPrecisionMode(Enum):
    """Precision modes for TensorRT conversions."""

    HIERARCHY = "hierarchy"
    SINGLE = "single"
    MIXED = "mixed"


@dataclass
class ShapeTuple(DataObject):
    """Represents a set of shapes for a single binding in a profile.

    Args:
        min (Tuple[int]): The minimum shape that the profile will support.
        opt (Tuple[int]): The shape for which TensorRT will optimize the engine.
        max (Tuple[int]): The maximum shape that the profile will support.
    """

    min: Tuple[int, ...]
    opt: Tuple[int, ...]
    max: Tuple[int, ...]

    def __str__(self):
        """String representation."""
        return f"(min={self.min}, opt={self.opt}, max={self.max})"

    def __repr__(self):
        """Representation."""
        return type(self).__name__ + self.__str__()

    def __iter__(self):
        """Iterate over shapes."""
        yield from [self.min, self.opt, self.max]


class MeasurementMode(Enum):
    """Measurement mode.

    `TIME_WINDOWS` mode run measurement windows with fixed time length.
    `COUNT_WINDOWS` mode run measurement windows with fixed number of requests.
    """

    TIME_WINDOWS = "time_windows"
    COUNT_WINDOWS = "count_windows"


@dataclass
class ProfilerConfig(DataObject):
    """Profiler configuration.

    For each batch size profiler will run measurments in windows. Depending on the measurement mode,
    each window will have fixed time length (MeasurementMode.TIME_WINDOWS)
    or fixed number of requests (MeasurementMode.COUNT_WINDOWS).
    Batch sizes are profiled in the ascending order.

    Profiler will run multiple trials and will stop when the measurements
    are stable (within `stability_percentage` from the mean) within three consecutive windows.
    If the measurements are not stable after `max_trials` trials, the profiler will stop with an error.
    Profiler will also stop profiling when the throughput does not increase at least by `throughput_cutoff_threshold`.


    Args:
        run_profiling (bool): If True, run profiling, otherwise skip profiling.
        batch_sizes (Optional[List[Union[int, None]]]): List of batch sizes to profile.
            None means that the model does not support batching.
        measurement_mode (MeasurementMode): Measurement mode.
        measurement_interval (Optional[float]): Measurement interval in milliseconds.
            Used only in MeasurementMode.TIME_WINDOWS mode.
        measurement_request_count (Optional[int]): Number of requests to measure in each window.
            Used only in MeasurementMode.COUNT_WINDOWS mode.
        stability_percentage (float): Allowed percentage of variation from the mean in three consecutive windows.
        max_trials (int): Maximum number of window trials.
        throughput_cutoff_threshold (float): Minimum throughput increase to continue profiling.
    """

    run_profiling: bool = True
    batch_sizes: Optional[List[Union[int, None]]] = None
    measurement_mode: MeasurementMode = MeasurementMode.COUNT_WINDOWS
    measurement_interval: Optional[float] = 5000  # ms
    measurement_request_count: Optional[int] = 50
    stability_percentage: float = 10.0
    max_trials: int = 10
    throughput_cutoff_threshold: float = DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD

    @classmethod
    def from_dict(cls, profiler_config_dict: Mapping) -> "ProfilerConfig":
        """Instantiate ProfilerConfig class from a dictionary.

        Args:
            profiler_config_dict (Mapping): Data dictionary.

        Returns:
            ProfilerConfig
        """
        return cls(
            run_profiling=profiler_config_dict.get("run_profiling", True),
            batch_sizes=profiler_config_dict.get("batch_sizes"),
            measurement_interval=profiler_config_dict.get("measurement_interval"),
            measurement_mode=MeasurementMode(
                profiler_config_dict.get("measurement_mode", MeasurementMode.TIME_WINDOWS)
            ),
            measurement_request_count=profiler_config_dict.get("measurement_request_count"),
            stability_percentage=profiler_config_dict.get("stability_percentage", 10.0),
            max_trials=profiler_config_dict.get("max_trials", 10),
            throughput_cutoff_threshold=profiler_config_dict.get("throughput_cutoff_threshold", -2),
        )


class TensorRTProfile(Dict[str, ShapeTuple]):
    """Single optimization profile that can be used to build an engine.

    More specifically, it is an ``Dict[str, ShapeTuple]`` which maps binding
    names to a set of min/opt/max shapes.
    """

    def add(self, name, min, opt, max):
        """A convenience function to add shapes for a single binding.

        Args:
            name (str): The name of the binding.
            min (Tuple[int]): The minimum shape that the profile will support.
            opt (Tuple[int]): The shape for which TensorRT will optimize the engine.
            max (Tuple[int]): The maximum shape that the profile will support.

        Returns:
            Profile:
                self, which allows this function to be easily chained to add multiple bindings,
                e.g., TensorRTProfile().add(...).add(...)
        """
        self[name] = ShapeTuple(min, opt, max)
        return self

    @classmethod
    def from_dict(cls, profile_dict: Dict[str, Dict[str, Tuple[int, ...]]]):
        """Create a TensorRTProfile from a dictionary.

        Args:
            profile_dict (Dict[str, Dict[str, Tuple[int, ...]]]):
                A dictionary mapping binding names to a dictionary containing ``min``, ``opt``, and
                ``max`` keys.

        Returns:
            TensorRTProfile:
                A TensorRTProfile object.
        """
        return cls({name: ShapeTuple(**shapes) for name, shapes in profile_dict.items()})

    def __getitem__(self, key):
        """Retrieves the shapes registered for a given input name.

        Returns:
            ShapeTuple:
                    A named tuple including ``min``, ``opt``, and ``max`` members for the shapes
                    corresponding to the input.
        """
        if key not in self:
            LOGGER.error(f"Binding: {key} does not have shapes set in this profile")
        return super().__getitem__(key)

    def __repr__(self):
        """Representation."""
        ret = "TensorRTProfile()"
        for name, (min, opt, max) in self.items():
            ret += f".add('{name}', min={min}, opt={opt}, max={max})"
        return ret

    def __str__(self):
        """String representation."""
        elems = []
        for name, (min, opt, max) in self.items():
            elems.append(f"{name} [min={min}, opt={opt}, max={max}]")

        sep = ",\n "
        return "{" + sep.join(elems) + "}"


SERIALIZED_FORMATS = (
    Format.TORCHSCRIPT,
    Format.TF_SAVEDMODEL,
    Format.TF_TRT,
    Format.TORCH_TRT,
    Format.ONNX,
    Format.TENSORRT,
)

SOURCE_FORMATS = (
    Format.TORCH,
    Format.TENSORFLOW,
    Format.JAX,
)

INPUT_FORMATS = {
    Framework.JAX: Format.JAX,
    Framework.TORCH: Format.TORCH,
    Framework.TENSORFLOW: Format.TENSORFLOW,
    Framework.ONNX: Format.ONNX,
}

EXPORT_FORMATS = {
    Framework.JAX: [Format.TF_SAVEDMODEL],
    Framework.TENSORFLOW: [Format.TF_SAVEDMODEL],
    Framework.TORCH: [Format.TORCHSCRIPT, Format.ONNX],
    Framework.ONNX: [Format.ONNX],
}

DEFAULT_JAX_TARGET_FORMATS = (
    Format.TF_SAVEDMODEL,
    Format.ONNX,
    Format.TENSORRT,
    Format.TF_TRT,
)

DEFAULT_TENSORFLOW_TARGET_FORMATS = (
    Format.TF_SAVEDMODEL,
    Format.TF_TRT,
    Format.ONNX,
    Format.TENSORRT,
)

DEFAULT_TORCH_TARGET_FORMATS = (
    Format.TORCHSCRIPT,
    Format.ONNX,
    Format.TORCH_TRT,
    Format.TENSORRT,
)

DEFAULT_ONNX_TARGET_FORMATS = (
    Format.ONNX,
    Format.TENSORRT,
)

DEFAULT_TARGET_FORMATS = {
    Framework.JAX: DEFAULT_JAX_TARGET_FORMATS,
    Framework.TENSORFLOW: DEFAULT_TENSORFLOW_TARGET_FORMATS,
    Framework.TORCH: DEFAULT_TORCH_TARGET_FORMATS,
    Framework.ONNX: DEFAULT_ONNX_TARGET_FORMATS,
}


AVAILABLE_JAX_TARGET_FORMATS = (Format.JAX,) + DEFAULT_JAX_TARGET_FORMATS

AVAILABLE_TENSORFLOW_TARGET_FORMATS = (Format.TENSORFLOW,) + DEFAULT_TENSORFLOW_TARGET_FORMATS

AVAILABLE_TORCH_TARGET_FORMATS = (Format.TORCH,) + DEFAULT_TORCH_TARGET_FORMATS

AVAILABLE_ONNX_TARGET_FORMATS = DEFAULT_ONNX_TARGET_FORMATS

AVAILABLE_TARGET_FORMATS = {
    Framework.JAX: AVAILABLE_JAX_TARGET_FORMATS,
    Framework.TENSORFLOW: AVAILABLE_TENSORFLOW_TARGET_FORMATS,
    Framework.TORCH: AVAILABLE_TORCH_TARGET_FORMATS,
    Framework.ONNX: AVAILABLE_ONNX_TARGET_FORMATS,
}

DEFAULT_TENSORRT_PRECISION = (TensorRTPrecision.FP32, TensorRTPrecision.FP16)
DEFAULT_TENSORRT_PRECISION_MODE = TensorRTPrecisionMode.HIERARCHY


class CustomConfig(abc.ABC):
    """Base class used for custom configs. Input for Model Navigator `optimize` method."""

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Name of the CustomConfig."""
        raise NotImplementedError()

    def defaults(self) -> None:
        """Update parameters to defaults."""
        return None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CustomConfig":
        """Instantiate CustomConfig from a dictionary."""
        return cls(**config_dict)


class CustomConfigForFormat(DataObject, CustomConfig):
    """Abstract base class used for custom configs representing particular format."""

    @property
    @abc.abstractmethod
    def format(self) -> Format:
        """Format represented by CustomConfig."""
        raise NotImplementedError()


@dataclass
class TensorFlowConfig(CustomConfigForFormat):
    """TensorFlow custom config used for SavedModel export.

    Args:
        jit_compile: Enable or Disable jit_compile flag for tf.function wrapper for Jax infer function.
        enable_xla: Enable or Disable enable_xla flag for jax2tf converter.

    """

    jit_compile: Tuple[Optional[bool], ...] = (None,)
    enable_xla: Tuple[Optional[bool], ...] = (None,)

    @property
    def format(self) -> Format:
        """Format represented by CustomConfig.

        Returns:
            TensorFlowConfig format
        """
        return Format.TF_SAVEDMODEL

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TensorFlow"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.jit_compile = (None,)
        self.enable_xla = (None,)


@dataclass
class TensorFlowTensorRTConfig(CustomConfigForFormat):
    """TensorFlow TensorRT custom config used for TensorRT SavedModel export.

    Args:
        precision: TensorRT precision.
        max_workspace_size: Max workspace size used by converter.
        minimum_segment_size: Min size of subgraph.
        trt_profile: TensorRT profile.
    """

    precision: Union[
        Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]
    ] = DEFAULT_TENSORRT_PRECISION
    max_workspace_size: Optional[int] = DEFAULT_MAX_WORKSPACE_SIZE
    minimum_segment_size: int = DEFAULT_MIN_SEGMENT_SIZE
    trt_profile: Optional[TensorRTProfile] = None

    def __post_init__(self) -> None:
        """Parse dataclass enums."""
        precision = (self.precision,) if not isinstance(self.precision, (list, tuple)) else self.precision
        self.precision = tuple(TensorRTPrecision(p) for p in precision)

    @property
    def format(self) -> Format:
        """Format represented by CustomConfig.

        Returns:
            TensorFlowTensorRTConfig format
        """
        return Format.TF_TRT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TensorFlowTensorRT"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.precision = tuple(TensorRTPrecision(p) for p in DEFAULT_TENSORRT_PRECISION)
        self.max_workspace_size = DEFAULT_MAX_WORKSPACE_SIZE
        self.minimum_segment_size = DEFAULT_MIN_SEGMENT_SIZE
        self.trt_profile = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TensorFlowTensorRTConfig":
        """Instantiate TensorFlowTensorRTConfig from  adictionary."""
        if config_dict.get("trt_profile") is not None and not isinstance(config_dict["trt_profile"], TensorRTProfile):
            config_dict["trt_profile"] = TensorRTProfile.from_dict(config_dict["trt_profile"])
        return cls(**config_dict)


@dataclass
class TorchConfig(CustomConfigForFormat):
    """Torch custom config used for TorchScript models export.

    Args:
        jit_type: Type of TorchScript export.

    """

    jit_type: Union[Union[str, JitType], Tuple[Union[str, JitType], ...]] = (JitType.SCRIPT, JitType.TRACE)

    def __post_init__(self) -> None:
        """Parse dataclass enums."""
        jit_type = (self.jit_type,) if not isinstance(self.jit_type, (list, tuple)) else self.jit_type
        self.jit_type = tuple(JitType(j) for j in jit_type)

    @property
    def format(self) -> Format:
        """Format represented by CustomConfig.

        Returns:
            TorchConfig format
        """
        return Format.TORCHSCRIPT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "Torch"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.jit_type = (JitType.SCRIPT, JitType.TRACE)


@dataclass
class TorchTensorRTConfig(CustomConfigForFormat):
    """Torch custom config used for TensorRT TorchScript conversion.

    Args:
        precision: TensorRT precision.
        max_workspace_size: Max workspace size used by converter.
        trt_profile: TensorRT profile.
    """

    precision: Union[
        Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]
    ] = DEFAULT_TENSORRT_PRECISION
    precision_mode: Optional[Union[str, TensorRTPrecisionMode]] = DEFAULT_TENSORRT_PRECISION_MODE
    trt_profile: Optional[TensorRTProfile] = None
    max_workspace_size: Optional[int] = DEFAULT_MAX_WORKSPACE_SIZE

    def __post_init__(self) -> None:
        """Parse dataclass enums."""
        precision = (self.precision,) if not isinstance(self.precision, (list, tuple)) else self.precision
        self.precision = tuple(TensorRTPrecision(p) for p in precision)
        self.precision_mode = TensorRTPrecisionMode(self.precision_mode)

    @property
    def format(self) -> Format:
        """Format represented by CustomConfig.

        Returns:
            TorchTensorRTConfig format
        """
        return Format.TORCH_TRT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TorchTensorRT"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.precision = tuple(TensorRTPrecision(p) for p in DEFAULT_TENSORRT_PRECISION)
        self.precision_mode = DEFAULT_TENSORRT_PRECISION_MODE
        self.trt_profile = None
        self.max_workspace_size = DEFAULT_MAX_WORKSPACE_SIZE

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TorchTensorRTConfig":
        """Instantiate TorchTensorRTConfig from  adictionary."""
        if config_dict.get("trt_profile") is not None and not isinstance(config_dict["trt_profile"], TensorRTProfile):
            config_dict["trt_profile"] = TensorRTProfile.from_dict(config_dict["trt_profile"])
        return cls(**config_dict)


@dataclass
class OnnxConfig(CustomConfigForFormat):
    """ONNX custom config used for ONNX export and conversion.

    Args:
        opset: ONNX opset used for conversion.
        dynamic_axes: Dynamic axes for ONNX conversion.
        onnx_extended_conversion: Enables additional conversions from TorchScript to ONNX.

    """

    opset: Optional[int] = DEFAULT_ONNX_OPSET
    dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None
    onnx_extended_conversion: bool = False

    @property
    def format(self) -> Format:
        """Format represented by CustomConfig.

        Returns:
            OnnxConfig format
        """
        return Format.ONNX

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "Onnx"


@dataclass
class TensorRTConfig(CustomConfigForFormat):
    """TensorRT custom config used for TensorRT conversion.

    Args:
        precision: TensorRT precision.
        max_workspace_size: Max workspace size used by converter.
        trt_profile: TensorRT profile.

    """

    precision: Union[
        Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]
    ] = DEFAULT_TENSORRT_PRECISION
    precision_mode: Union[str, TensorRTPrecisionMode] = TensorRTPrecisionMode.HIERARCHY
    trt_profile: Optional[TensorRTProfile] = None
    max_workspace_size: Optional[int] = DEFAULT_MAX_WORKSPACE_SIZE

    def __post_init__(self) -> None:
        """Parse dataclass enums."""
        self.precision_mode = TensorRTPrecisionMode(self.precision_mode)
        precision = (self.precision,) if not isinstance(self.precision, (list, tuple)) else self.precision
        self.precision = tuple(TensorRTPrecision(p) for p in precision)

    @property
    def format(self) -> Format:
        """Format represented by CustomConfig.

        Returns:
            TensorRTConfig format
        """
        return Format.TENSORRT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TensorRT"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.precision = tuple(TensorRTPrecision(p) for p in DEFAULT_TENSORRT_PRECISION)
        self.precision_mode = DEFAULT_TENSORRT_PRECISION_MODE
        self.trt_profile = None
        self.max_workspace_size = DEFAULT_MAX_WORKSPACE_SIZE

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TensorRTConfig":
        """Instantiate TensorRTConfig from  adictionary."""
        if config_dict.get("trt_profile") is not None and not isinstance(config_dict["trt_profile"], TensorRTProfile):
            config_dict["trt_profile"] = TensorRTProfile.from_dict(config_dict["trt_profile"])
        return cls(**config_dict)


def map_custom_configs(custom_configs: Optional[Sequence[CustomConfig]]) -> Dict:
    """Map custom configs from list to dictionary.

    Args:
        custom_configs: List of custom configs passed to API method

    Returns:
        Mapped configs to dictionary
    """
    if not custom_configs:
        return {}

    return {config.name(): config for config in custom_configs}


def _custom_configs() -> Dict[str, Type[CustomConfigForFormat]]:
    custom_configs = {}
    custom_configs_formats = {}
    for cls in CustomConfigForFormat.__subclasses__():
        assert cls.name() not in custom_configs
        cls_format = cls().format
        assert cls_format not in custom_configs_formats

        custom_configs_formats[cls_format] = custom_configs_formats
        custom_configs[cls.name()] = cls

    return custom_configs


CUSTOM_CONFIGS_MAPPING = _custom_configs()
