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
"""Definition of enums and classes representing configuration for Model Navigator."""

import abc
import copy
import dataclasses
import inspect
import itertools
import pathlib
import warnings
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

from model_navigator.configuration.constants import (
    DEFAULT_MAX_TRIALS,
    DEFAULT_MAX_WORKSPACE_SIZE,
    DEFAULT_MAX_WORKSPACE_SIZE_TFTRT,
    DEFAULT_MAX_WORKSPACE_SIZE_TORCHTRT,
    DEFAULT_MIN_SEGMENT_SIZE,
    DEFAULT_MIN_TRIALS,
    DEFAULT_ONNX_OPSET,
    DEFAULT_PICKLE_PROTOCOL_TORCHTRT,
    DEFAULT_STABILITY_PERCENTAGE,
    DEFAULT_STABILIZATION_WINDOWS,
    DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
)
from model_navigator.core.logger import LOGGER
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.frameworks import Framework
from model_navigator.utils.common import DataObject

Sample = Dict[str, np.ndarray]

VerifyFunction = Callable[[Iterable[Sample], Iterable[Sample]], bool]


class DeviceKind(Enum):
    """Supported types of devices.

    Args:
        CPU (str): Select CPU device.
        GPU (str): Select GPU with CUDA support.
    """

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
    """All model formats supported by Model Navigator 'optimize' function.

    Args:
        PYTHON (str): Format indicating any model defined in Python.
        TORCH (str): Format indicating PyTorch model.
        TENSORFLOW (str): Format indicating TensorFlow model.
        JAX (str): Format indicating JAX model.
        TORCHSCRIPT (str): Format indicating TorchScript model.
        TF_SAVEDMODEL (str): Format indicating TensorFlow SavedModel.
        TF_TRT (str): Format indicating TensorFlow TensorRT model.
        TORCH_TRT (str): Format indicating PyTorch TensorRT model.
        ONNX (str): Format indicating ONNX model.
        TENSORRT (str): Format indicating TensorRT model.
    """

    PYTHON = "python"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    TORCHSCRIPT = "torchscript"
    TORCH_EXPORTEDPROGRAM = "torch-exportedprogram"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TORCH_TRT = "torch-trt"
    ONNX = "onnx"
    TENSORRT = "trt"


class JitType(Enum):
    """TorchScript export parameter.

    Used for selecting the type of TorchScript export.

    Args:
        TRACE (str): Use tracing during export.
        SCRIPT (str): Use scripting during export.
    """

    SCRIPT = "script"
    TRACE = "trace"


class AutocastType(Enum):
    """Torch runner autocast options.

    Args:
        DEVICE: Use device default dtype.
        FP16 (str): Use float16 autocast during runtime.
        BF16 (str): Use bfloat16 autocast during runtime.
    """

    DEVICE = None
    FP16 = "torch.float16"
    BF16 = "torch.bfloat16"


PrecisionType = Literal["int8", "fp8", "fp16", "bf16", "fp32"]


class TensorRTPrecision(Enum):
    """Precisions supported during TensorRT conversions.

    Args:
        INT8 (str): 8-bit integer precision.
        FP8 (str): 8-bit floating point precision.
        FP16 (str): 16-bit floating point precision.
        BF16 (str): 16-bit brain floating point precision.
        FP32 (str): 32-bit floating point precision.
        NVFP4 (str): 4-bit floating point precision.
    """

    INT8 = "int8"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    NVFP4 = "nvfp4"


class TensorRTPrecisionMode(Enum):
    """Precision modes for TensorRT conversions.

    Args:
        HIERARCHY (str): Use TensorRT precision hierarchy starting from highest to lowest.
        SINGLE (str): Use single precision.
        MIXED (str): Use mixed precision.
    """

    HIERARCHY = "hierarchy"
    SINGLE = "single"
    MIXED = "mixed"


class TensorType(Enum):
    """All model formats supported by Model Navigator 'optimize' function."""

    NUMPY = "numpy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class TensorRTCompatibilityLevel(Enum):
    """Compatibility level for TensorRT.

    Args:
        AMPERE_PLUS (str): Support AMPERE plus architecture
    """

    AMPERE_PLUS = "ampere_plus"


@dataclasses.dataclass
class ShapeTuple(DataObject):
    """Represents a set of shapes for a single binding in a profile.

    Each element of the tuple represents a shape for a single dimension of the binding.

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


@dataclasses.dataclass
class OptimizationProfile(DataObject):
    """Optimization profile configuration.

    For each batch size profiler will run measurements in windows of fixed number of queries.
    Batch sizes are profiled in the ascending order.

    Profiler will run multiple trials and will stop when the measurements
    are stable (within `stability_percentage` from the mean) within three consecutive windows.
    If the measurements are not stable after `max_trials` trials, the profiler will stop with an error.
    Profiler will also stop profiling when the throughput does not increase at least by `throughput_cutoff_threshold`.


    Args:
        max_batch_size: Maximal batch size used during conversion and profiling. None mean automatic search is enabled.
        batch_sizes : List of batch sizes to profile. None mean automatic search is enabled.
        window_size: Number of requests to measure in each window.
        stability_percentage: Allowed percentage of variation from the mean in consecutive windows.
        stabilization_windows: Number consecutive windows selected for stabilization.
        min_trials: Minimal number of window trials.
        max_trials: Maximum number of window trials.
        throughput_cutoff_threshold: Minimum throughput increase to continue profiling.
        throughput_backoff_limit: Back-off limit to run multiple more profiling steps to avoid stop at local minimum
                                  when throughput saturate based on `throughput_cutoff_threshold`.
        dataloader: Optional dataloader for profiling. Use only 1 sample.
    """

    max_batch_size: Optional[int] = None
    batch_sizes: Optional[List[Union[int, None]]] = None
    window_size: int = DEFAULT_WINDOW_SIZE
    stability_percentage: float = DEFAULT_STABILITY_PERCENTAGE
    stabilization_windows: int = DEFAULT_STABILIZATION_WINDOWS
    min_trials: int = DEFAULT_MIN_TRIALS
    max_trials: int = DEFAULT_MAX_TRIALS
    throughput_cutoff_threshold: Optional[float] = DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD
    throughput_backoff_limit: int = DEFAULT_THROUGHPUT_BACKOFF_LIMIT
    dataloader: Optional[SizedDataLoader] = None

    def __post_init__(self):
        """Validate OptimizationProfile definition to avoid unsupported configurations."""
        if self.stability_percentage <= 0:
            raise ModelNavigatorConfigurationError("`stability_percentage` must be greater than 0.0.")

        if self.throughput_backoff_limit < 0:
            raise ModelNavigatorConfigurationError("`throughput_backoff_limit` must be greater then or equal to 0.")

        greater_or_equal_1 = [
            "window_size",
            "stability_percentage",
            "stabilization_windows",
            "min_trials",
            "max_trials",
        ]
        for member in greater_or_equal_1:
            value = getattr(self, member)
            if value < 1:
                raise ModelNavigatorConfigurationError(f"`{member}` must be greater or equal 1.")

        if self.min_trials < self.stabilization_windows:
            raise ModelNavigatorConfigurationError(
                "`min_trials` must be greater or equal than `stabilization_windows`."
            )

        if self.min_trials > self.max_trials:
            raise ModelNavigatorConfigurationError("`max_trials` must be greater or equal `min_trials`.")

    def to_dict(self, filter_fields: Optional[List[str]] = None, parse: bool = False) -> Dict:
        """Serialize to a dictionary.

        Append `dataloader` field to filtered fields during dump.

        Args:
            filter_fields (Optional[List[str]], optional): List of fields to filter out.
                Defaults to None.
            parse (bool, optional): If True recursively parse field values to jsonable representation.
                Defaults to False.

        Returns:
            Dict: Data serialized to a dictionary.
        """
        if not filter_fields:
            filter_fields = []

        filter_fields += ["dataloader"]
        return super().to_dict(filter_fields=filter_fields, parse=parse)

    @classmethod
    def from_dict(cls, optimization_profile_dict: Mapping) -> "OptimizationProfile":
        """Instantiate OptimizationProfile class from a dictionary.

        Args:
            optimization_profile_dict (Mapping): Data dictionary.

        Returns:
            OptimizationProfile
        """
        return cls(
            max_batch_size=optimization_profile_dict.get("max_batch_size"),
            batch_sizes=optimization_profile_dict.get("batch_sizes"),
            window_size=optimization_profile_dict.get("window_size", DEFAULT_WINDOW_SIZE),
            stability_percentage=optimization_profile_dict.get("stability_percentage", DEFAULT_STABILITY_PERCENTAGE),
            stabilization_windows=optimization_profile_dict.get("stabilization_windows", DEFAULT_STABILIZATION_WINDOWS),
            min_trials=optimization_profile_dict.get("min_trials", DEFAULT_MIN_TRIALS),
            max_trials=optimization_profile_dict.get("max_trials", DEFAULT_MAX_TRIALS),
            throughput_cutoff_threshold=optimization_profile_dict.get(
                "throughput_cutoff_threshold", DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD
            ),
            throughput_backoff_limit=optimization_profile_dict.get(
                "throughput_backoff_limit", DEFAULT_THROUGHPUT_BACKOFF_LIMIT
            ),
        )

    def clone(self) -> "OptimizationProfile":
        """Clone the current OptimizationProfile using deepcopy."""
        return copy.deepcopy(self)


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

    def to_dict(self) -> Dict[str, Dict[str, Tuple[int, ...]]]:
        """Serialize to a dictionary.

        Returns:
            Dict[str, Dict[str, Tuple[int, ...]]]:
                A dictionary mapping binding names to a dictionary containing ``min``, ``opt``, and
                ``max`` keys.
        """
        return {name: vars(shapes) for name, shapes in self.items()}

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
    Framework.NONE: Format.PYTHON,
    Framework.JAX: Format.JAX,
    Framework.TORCH: Format.TORCH,
    Framework.TENSORFLOW: Format.TENSORFLOW,
    Framework.ONNX: Format.ONNX,
    Framework.TENSORRT: Format.TENSORRT,
}

EXPORT_FORMATS = {
    Framework.NONE: [],
    Framework.JAX: [Format.TF_SAVEDMODEL],
    Framework.TENSORFLOW: [Format.TF_SAVEDMODEL],
    Framework.TORCH: [Format.TORCHSCRIPT, Format.ONNX, Format.TORCH_EXPORTEDPROGRAM],
    Framework.ONNX: [Format.ONNX],
    Framework.TENSORRT: [Format.TENSORRT],
}

DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS = (Format.PYTHON,)

DEFAULT_JAX_TARGET_FORMATS = (
    Format.TF_SAVEDMODEL,
    Format.ONNX,
    Format.TENSORRT,
)

DEFAULT_TENSORFLOW_TARGET_FORMATS = (
    Format.TF_SAVEDMODEL,
    Format.ONNX,
    Format.TENSORRT,
)

DEFAULT_TORCH_TARGET_FORMATS = (
    Format.TORCHSCRIPT,
    Format.TORCH_EXPORTEDPROGRAM,
    Format.ONNX,
    Format.TORCH_TRT,
    Format.TENSORRT,
)

DEFAULT_TORCH_TARGET_FORMATS_FOR_PROFILING = (
    Format.TORCH,
    Format.TORCHSCRIPT,
    Format.TORCH_EXPORTEDPROGRAM,
    Format.ONNX,
    Format.TORCH_TRT,
    Format.TENSORRT,
)

DEFAULT_ONNX_TARGET_FORMATS = (
    Format.ONNX,
    Format.TENSORRT,
)

DEFAULT_TENSORRT_TARGET_FORMATS = (Format.TENSORRT,)

DEFAULT_TARGET_FORMATS = {
    Framework.NONE: DEFAULT_NONE_FRAMEWORK_TARGET_FORMATS,
    Framework.JAX: DEFAULT_JAX_TARGET_FORMATS,
    Framework.TENSORFLOW: DEFAULT_TENSORFLOW_TARGET_FORMATS,
    Framework.TORCH: DEFAULT_TORCH_TARGET_FORMATS,
    Framework.ONNX: DEFAULT_ONNX_TARGET_FORMATS,
    Framework.TENSORRT: DEFAULT_TENSORRT_TARGET_FORMATS,
}

AVAILABLE_JAX_TARGET_FORMATS = (Format.JAX,) + DEFAULT_JAX_TARGET_FORMATS

AVAILABLE_TENSORFLOW_TARGET_FORMATS = (Format.TENSORFLOW,) + DEFAULT_TENSORFLOW_TARGET_FORMATS

AVAILABLE_TORCH_TARGET_FORMATS = (Format.TORCH,) + DEFAULT_TORCH_TARGET_FORMATS

AVAILABLE_ONNX_TARGET_FORMATS = DEFAULT_ONNX_TARGET_FORMATS

AVAILABLE_NONE_FRAMEWORK_TARGET_FORMATS = (Format.PYTHON,)

AVAILABLE_TENSORRT_TARGET_FORMATS = (Format.TENSORRT,)

AVAILABLE_TARGET_FORMATS = {
    Framework.NONE: AVAILABLE_NONE_FRAMEWORK_TARGET_FORMATS,
    Framework.JAX: AVAILABLE_JAX_TARGET_FORMATS,
    Framework.TENSORFLOW: AVAILABLE_TENSORFLOW_TARGET_FORMATS,
    Framework.TORCH: AVAILABLE_TORCH_TARGET_FORMATS,
    Framework.ONNX: AVAILABLE_ONNX_TARGET_FORMATS,
    Framework.TENSORRT: AVAILABLE_TENSORRT_TARGET_FORMATS,
}

DEFAULT_TENSORRT_PRECISION = (
    TensorRTPrecision.FP32.value,
    TensorRTPrecision.FP16.value,
)
DEFAULT_TENSORRT_PRECISION_MODE = TensorRTPrecisionMode.HIERARCHY.value

TensorRTPrecisionType = Union[Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]]
TensorRTPrecisionModeType = Union[str, TensorRTPrecisionMode]


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


@dataclasses.dataclass
class CustomConfigForFormat(DataObject, CustomConfig):
    """Abstract base class used for custom configs representing particular format.

    Args:
        custom_args: Custom arguments passed to conversion function.
        device: torch-like string used for selecting device e.q. "cuda:0".
    """

    custom_args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    device: Optional[str] = None

    @property
    @abc.abstractmethod
    def format(self) -> Format:
        """Format represented by CustomConfig."""
        raise NotImplementedError()


@dataclasses.dataclass
class CustomConfigForTensorRT(CustomConfigForFormat):
    """Abstract base class used for custom configs representing particular TensorRT format."""

    trt_profiles: Optional[List[TensorRTProfile]] = None
    trt_profile: Optional[TensorRTProfile] = None  # TODO: Remove before 1.0.0 release
    precision: TensorRTPrecisionType = DEFAULT_TENSORRT_PRECISION
    precision_mode: TensorRTPrecisionModeType = DEFAULT_TENSORRT_PRECISION_MODE
    max_workspace_size: Optional[int] = DEFAULT_MAX_WORKSPACE_SIZE
    conversion_fallback: bool = False

    def __post_init__(self):
        """Initialize common TensorRT parameters and validate configuration."""
        precision = (self.precision,) if not isinstance(self.precision, (list, tuple)) else self.precision
        self.precision = tuple(TensorRTPrecision(p) for p in precision)
        self.precision_mode = TensorRTPrecisionMode(self.precision_mode)

        # TODO: Remove before 1.0.0 release
        if self.trt_profile is not None and self.trt_profiles is not None:
            raise ModelNavigatorConfigurationError("Only one of trt_profile and trt_profiles can be set.")
        elif self.trt_profile:
            warnings.warn(
                "trt_profile will be deprecated in future releases. Use trt_profiles instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            self.trt_profiles = [self.trt_profile]

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.precision = tuple(TensorRTPrecision(p) for p in DEFAULT_TENSORRT_PRECISION)
        self.precision_mode = TensorRTPrecisionMode(DEFAULT_TENSORRT_PRECISION_MODE)
        self.max_workspace_size = DEFAULT_MAX_WORKSPACE_SIZE
        self.trt_profiles = None
        self.trt_profile = None
        self.conversion_fallback = False


@dataclasses.dataclass
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
        """Returns Format.TF_SAVEDMODEL.

        Returns:
            Format.TF_SAVEDMODEL
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


@dataclasses.dataclass
class TensorFlowTensorRTConfig(CustomConfigForTensorRT):
    """TensorFlow TensorRT custom config used for TensorRT SavedModel export.

    Args:
        minimum_segment_size: Min size of subgraph.
    """

    minimum_segment_size: int = DEFAULT_MIN_SEGMENT_SIZE
    max_workspace_size: Optional[int] = DEFAULT_MAX_WORKSPACE_SIZE_TFTRT

    @property
    def format(self) -> Format:
        """Returns Format.TF_TRT.

        Returns:
            Format.TF_TRT
        """
        return Format.TF_TRT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TensorFlowTensorRT"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        super().defaults()
        self.minimum_segment_size = DEFAULT_MIN_SEGMENT_SIZE

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TensorFlowTensorRTConfig":
        """Instantiate TensorFlowTensorRTConfig from a dictionary."""
        if config_dict.get("trt_profiles") is not None:
            # if config_dict.get("trt_profiles") is not None and not isinstance(config_dict["trt_profile"], TensorRTProfile):
            parsed_trt_profiles = []
            for trt_profile in config_dict.get("trt_profiles"):
                if not isinstance(trt_profile, TensorRTProfile):
                    trt_profile = TensorRTProfile.from_dict(trt_profile)
                parsed_trt_profiles.append(trt_profile)
            config_dict["trt_profiles"] = parsed_trt_profiles
        return cls(**config_dict)


@dataclasses.dataclass
class TorchConfig(CustomConfigForFormat):
    """Torch custom config used for torch runner.

    Args:
        autocast: Enable Automatic Mixed Precision in runner (default: False).
        autocast_dtype: dtype used for autocast
        inference_mode: Enable inference mode in runner (default: True).
    """

    autocast: bool = True
    autocast_dtype: AutocastType = AutocastType.DEVICE
    inference_mode: bool = True
    custom_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post initialization to handle correctly enums."""
        self.autocast_dtype: AutocastType = AutocastType(self.autocast_dtype)

    @property
    def format(self) -> Format:
        """Returns Format.TORCH.

        Returns:
            Format.TORCH
        """
        return Format.TORCH

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "Torch"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.autocast = True
        self.autocast_dtype = AutocastType.DEVICE
        self.inference_mode = True
        self.custom_args = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TorchConfig":
        """Instantiate TorchConfig from a dictionary."""
        return cls(
            autocast=config_dict.get("autocast", True),
            autocast_dtype=AutocastType(config_dict.get("autocast_dtype", AutocastType.DEVICE)),
            inference_mode=config_dict.get("inference_mode", True),
            custom_args=config_dict.get("custom_args"),
        )


@dataclasses.dataclass
class TorchScriptConfig(CustomConfigForFormat):
    """Torch custom config used for TorchScript export.

    Args:
        jit_type: Type of TorchScript export.
        strict: Enable or Disable strict flag for tracer used in TorchScript export (default: True).
        autocast: Enable Automatic Mixed Precision in runner (default: False).
        autocast_dtype: dtype used for autocast
        inference_mode: Enable inference mode in runner (default: True).
    """

    jit_type: Union[Union[str, JitType], Tuple[Union[str, JitType], ...]] = (JitType.SCRIPT, JitType.TRACE)
    strict: bool = True
    autocast: bool = True
    autocast_dtype: AutocastType = AutocastType.DEVICE
    inference_mode: bool = True

    def __post_init__(self) -> None:
        """Parse dataclass enums."""
        jit_type = (self.jit_type,) if not isinstance(self.jit_type, (list, tuple)) else self.jit_type
        self.jit_type = tuple(JitType(j) for j in jit_type)
        self.autocast_dtype: AutocastType = AutocastType(self.autocast_dtype)

    @property
    def format(self) -> Format:
        """Returns Format.TORCHSCRIPT.

        Returns:
            Format.TORCHSCRIPT
        """
        return Format.TORCHSCRIPT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TorchScript"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.jit_type = (JitType.SCRIPT, JitType.TRACE)
        self.strict = True
        self.autocast = True
        self.autocast_dtype = AutocastType.DEVICE
        self.inference_mode = True


@dataclasses.dataclass
class TorchExportConfig(CustomConfigForFormat):
    """Torch export custom config used for torch.export.export.

    Args:
        autocast: Enable Automatic Mixed Precision in runner (default: False).
        autocast_dtype: dtype used for autocast
        inference_mode: Enable inference mode in runner (default: True).
    """

    autocast: bool = True
    autocast_dtype: AutocastType = AutocastType.DEVICE
    inference_mode: bool = True

    def __post_init__(self):
        """Post initialization to handle correctly enums."""
        self.autocast_dtype: AutocastType = AutocastType(self.autocast_dtype)

    @property
    def format(self) -> Format:
        """Returns Format.

        Returns:
            Format.TORCH_EXPORTEDPROGRAM
        """
        return Format.TORCH_EXPORTEDPROGRAM

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TorchExport"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        self.autocast = True
        self.autocast_dtype = AutocastType.DEVICE
        self.inference_mode = True


@dataclasses.dataclass
class TorchTensorRTConfig(CustomConfigForTensorRT):
    """Torch custom config used for TensorRT TorchScript conversion."""

    max_workspace_size: Optional[int] = DEFAULT_MAX_WORKSPACE_SIZE_TORCHTRT
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL_TORCHTRT

    @property
    def format(self) -> Format:
        """Returns Format.TORCH_TRT.

        Returns:
            Format.TORCH_TRT
        """
        return Format.TORCH_TRT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TorchTensorRT"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TorchTensorRTConfig":
        """Instantiate TorchTensorRTConfig from a dictionary."""
        if config_dict.get("trt_profiles") is not None:
            # if config_dict.get("trt_profiles") is not None and not isinstance(config_dict["trt_profile"], TensorRTProfile):
            parsed_trt_profiles = []
            for trt_profile in config_dict.get("trt_profiles"):
                if not isinstance(trt_profile, TensorRTProfile):
                    trt_profile = TensorRTProfile.from_dict(trt_profile)
                parsed_trt_profiles.append(trt_profile)
            config_dict["trt_profiles"] = parsed_trt_profiles
        return cls(**config_dict)


@dataclasses.dataclass
class OnnxTraceExportConfig(DataObject):
    """ONNX export config used for ONNX Torch Trace export.

    Torch Trace export is performed by default, but when OnnxDynamoExportConfig
    is used in export_engins list OnnxTraceExportConfig must be explicitly
    provided to be performed.
    """


@dataclasses.dataclass
class OnnxDynamoExportConfig(DataObject):
    """ONNX export config used for ONNX Torch Dynamo export.

    Args:
        dynamo_dynamic_shapes: Enable dynamic shapes for dynamo export.
            By default dynamic shapes are enabled if dynamic_axes are set or batching is enabled.
    """

    dynamo_dynamic_shapes: Optional[bool] = None


OnnxExportEngineType = Union[OnnxTraceExportConfig, OnnxDynamoExportConfig]
_EngineTypeT = TypeVar("_EngineTypeT", bound=OnnxExportEngineType)


@dataclasses.dataclass
class OnnxConfig(CustomConfigForFormat):
    """ONNX custom config used for ONNX export and conversion.

    Args:
        opset: ONNX opset used for conversion.
        onnx_extended_conversion: Enables additional conversions from TorchScript to ONNX.
        graph_surgeon_optimization: Enables polygraphy graph surgeon optimization: fold_constants, infer_shapes, toposort, cleanup.
        export_device: Device used for ONNX export.
        dynamic_axes: Dynamic axes for ONNX conversion.
        model_path: optional path to onnx model file, if provided the model will be loaded from the file instead of exporting to onnx
        export_engine: List of export engines to use. Expects only one engine of a type. First of each type will be used.
            Currently, only Torch Dynamo exports engine is supported in addtion to default Torch Trace export.
    """

    opset: Optional[int] = DEFAULT_ONNX_OPSET
    onnx_extended_conversion: bool = False
    graph_surgeon_optimization: bool = True
    export_device: Optional[str] = None
    dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None
    model_path: Optional[Union[str, pathlib.Path]] = None
    export_engine: List[OnnxExportEngineType] = dataclasses.field(default_factory=lambda: [OnnxTraceExportConfig()])

    @property
    def format(self) -> Format:
        """Returns Format.ONNX.

        Returns:
            Format.ONNX
        """
        return Format.ONNX

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "Onnx"

    def defaults(self) -> None:
        """Update parameters to defaults.

        Only configuration related to ONNX export and conversion parameters are updated. We leave the dynamo and
        extended conversion flags as are set during config initialization.
        """
        super().defaults()
        self.opset = DEFAULT_ONNX_OPSET
        self.graph_surgeon_optimization = True
        self.export_device = None
        self.dynamic_axes = None
        self.model_path = None
        self.export_engine = [OnnxTraceExportConfig()]

    def get_export_engine(self, engine_type: Type[_EngineTypeT]) -> Optional[_EngineTypeT]:
        """Find given export engine in export_engine list.

        Args:
            engine_type: Type of export engine to find

        Returns:
            Export engine if found, otherwise None
        """
        for engine in self.export_engine:
            if isinstance(engine, engine_type):
                return engine
        return None


@dataclasses.dataclass
class TensorRTConfig(CustomConfigForTensorRT):
    """TensorRT custom config used for TensorRT conversion.

    Args:
        optimization_level: Optimization level for TensorRT conversion. Allowed values are fom 0 to 5. Where default is
                            3 based on TensorRT API documentation.
        compatibility_level: Compatibility level for TensorRT conversion.
        onnx_parser_flags: List of TensorRT OnnxParserFlags used for conversion.
        timing_cache_dir: Storage directory for TRT tactic timing info collected from builder
        model_path: optional path to trt model file, if provided the model will be loaded from the file instead of converting onnx to trt
    """

    optimization_level: Optional[int] = None
    compatibility_level: Optional[TensorRTCompatibilityLevel] = None
    onnx_parser_flags: Optional[List[int]] = None
    timing_cache_dir: Optional[Union[str, pathlib.Path]] = None
    model_path: Optional[Union[str, pathlib.Path]] = None

    def __post_init__(self) -> None:
        """Parse dataclass enums."""
        super().__post_init__()
        if self.optimization_level is not None and (self.optimization_level < 0 or self.optimization_level > 5):
            raise ModelNavigatorConfigurationError(
                f"TensorRT `optimization_level` must be between 0 and 5. Provided value: {self.optimization_level}."
            )

    @property
    def format(self) -> Format:
        """Returns Format.TENSORRT.

        Returns:
            Format.TENSORRT
        """
        return Format.TENSORRT

    @classmethod
    def name(cls) -> str:
        """Name of the config."""
        return "TensorRT"

    def defaults(self) -> None:
        """Update parameters to defaults."""
        super().defaults()
        self.optimization_level = None
        self.compatibility_level = None
        self.onnx_parser_flags = None
        self.timing_cache_dir = None
        self.model_path = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TensorRTConfig":
        """Instantiate TensorRTConfig from a dictionary."""
        if config_dict.get("trt_profiles") is not None:
            parsed_trt_profiles = []
            for trt_profile in config_dict.get("trt_profiles"):
                if not isinstance(trt_profile, TensorRTProfile):
                    trt_profile = TensorRTProfile.from_dict(trt_profile)
                parsed_trt_profiles.append(trt_profile)
            config_dict["trt_profiles"] = parsed_trt_profiles
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
    for cls in itertools.chain(CustomConfigForFormat.__subclasses__(), CustomConfigForTensorRT.__subclasses__()):
        if inspect.isabstract(cls):
            continue
        assert cls.name() not in custom_configs
        cls_format = cls().format
        assert cls_format not in custom_configs_formats

        custom_configs_formats[cls_format] = custom_configs_formats
        custom_configs[cls.name()] = cls

    return custom_configs


CUSTOM_CONFIGS_MAPPING = _custom_configs()


class RuntimeSearchStrategy:
    """Base class for runtime search strategies."""

    def __str__(self):
        """Return name of strategy."""
        return self.__class__.__name__


class MinLatencyStrategy(RuntimeSearchStrategy):
    """Get runtime with the lowest latency."""

    pass


class MaxThroughputStrategy(RuntimeSearchStrategy):
    """Get runtime with the highest throughput."""

    pass


class MaxThroughputAndMinLatencyStrategy(RuntimeSearchStrategy):
    """Get runtime with the highest throughput and the lowest latency."""

    pass


class MaxThroughputWithLatencyBudgetStrategy(RuntimeSearchStrategy):
    """Get runtime with the hightest throughput within the latency budget."""

    def __init__(self, latency_budget: float) -> None:
        """Initialize the class.

        Args:
            latency_budget: Latency budget in milliseconds.
        """
        super().__init__()
        self.latency_budget = latency_budget

    def __str__(self):
        """Return name of strategy."""
        return f"{self.__class__.__name__}({self.latency_budget}[ms])"


class SelectedRuntimeStrategy(RuntimeSearchStrategy):
    """Get a selected runtime."""

    def __init__(self, model_key: str, runner_name: str) -> None:
        """Initialize the class.

        Args:
            model_key (str): Unique key of the model.
            runner_name (str): Name of the runner.
        """
        super().__init__()
        self.model_key = model_key
        self.runner_name = runner_name

    def __str__(self):
        """Return name of strategy."""
        return f"{self.__class__.__name__}({self.model_key}:{self.runner_name})"


DEFAULT_RUNTIME_STRATEGIES = [MaxThroughputAndMinLatencyStrategy(), MinLatencyStrategy()]
