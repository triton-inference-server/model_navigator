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
"""Definition of enums and classes representing configuration for Model Navigator."""

import abc
import dataclasses
import inspect
import itertools
import warnings
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

import numpy as np

from model_navigator.core.constants import (
    DEFAULT_MAX_WORKSPACE_SIZE,
    DEFAULT_MIN_SEGMENT_SIZE,
    DEFAULT_ONNX_OPSET,
    DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD,
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


class TensorRTPrecision(Enum):
    """Precisions supported during TensorRT conversions.

    Args:
        INT8 (str): 8-bit integer precision.
        FP8 (str): 8-bit floating point precision.
        FP16 (str): 16-bit floating point precision.
        BF16 (str): 16-bit brain floating point precision.
        FP32 (str): 32-bit floating point precision.
    """

    INT8 = "int8"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


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
        dataloader: Optional dataloader for profiling. Use only 1 sample.
    """

    max_batch_size: Optional[int] = None
    batch_sizes: Optional[List[Union[int, None]]] = None
    window_size: int = 50
    stability_percentage: float = 10.0
    stabilization_windows: int = 3
    min_trials: int = 3
    max_trials: int = 10
    throughput_cutoff_threshold: float = DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD
    dataloader: Optional[SizedDataLoader] = None

    def __post_init__(self):
        """Validate OptimizationProfile definition to avoid unsupported configurations."""
        if self.stability_percentage <= 0:
            raise ModelNavigatorConfigurationError("`stability_percentage` must be greater than 0.0.")

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
            window_size=optimization_profile_dict.get("window_size", 50),
            stability_percentage=optimization_profile_dict.get("stability_percentage", 10.0),
            stabilization_windows=optimization_profile_dict.get("stabilization_windows", 3),
            min_trials=optimization_profile_dict.get("min_trials", 3),
            max_trials=optimization_profile_dict.get("max_trials", 10),
            throughput_cutoff_threshold=optimization_profile_dict.get(
                "throughput_cutoff_threshold", DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD
            ),
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
    TensorRTPrecision.FP32,
    TensorRTPrecision.FP16,
)
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
    precision: Union[Union[str, TensorRTPrecision], Tuple[Union[str, TensorRTPrecision], ...]] = (
        DEFAULT_TENSORRT_PRECISION
    )
    precision_mode: Optional[Union[str, TensorRTPrecisionMode]] = DEFAULT_TENSORRT_PRECISION_MODE
    max_workspace_size: Optional[int] = DEFAULT_MAX_WORKSPACE_SIZE
    run_max_batch_size_search: Optional[bool] = None  # TODO this parameter is currently not used

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
        self.precision_mode = DEFAULT_TENSORRT_PRECISION_MODE
        self.max_workspace_size = DEFAULT_MAX_WORKSPACE_SIZE
        self.trt_profiles = None
        self.trt_profile = None
        self.run_max_batch_size_search = None


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
        inference_mode: Enable inference mode in runner (default: True).
    """

    autocast: bool = False
    inference_mode: bool = True

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
        self.autocast = False
        self.inference_mode = True


@dataclasses.dataclass
class TorchScriptConfig(CustomConfigForFormat):
    """Torch custom config used for TorchScript export.

    Args:
        jit_type: Type of TorchScript export.
        strict: Enable or Disable strict flag for tracer used in TorchScript export (default: True).
        autocast: Enable Automatic Mixed Precision in runner (default: False).
        inference_mode: Enable inference mode in runner (default: True).
    """

    jit_type: Union[Union[str, JitType], Tuple[Union[str, JitType], ...]] = (JitType.SCRIPT, JitType.TRACE)
    strict: bool = True
    autocast: bool = False
    inference_mode: bool = True

    def __post_init__(self) -> None:
        """Parse dataclass enums."""
        jit_type = (self.jit_type,) if not isinstance(self.jit_type, (list, tuple)) else self.jit_type
        self.jit_type = tuple(JitType(j) for j in jit_type)

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
        self.autocast = False
        self.inference_mode = True


@dataclasses.dataclass
class TorchExportConfig(CustomConfigForFormat):
    """Torch export custom config used for torch.export.export.

    Args:
        autocast: Enable Automatic Mixed Precision in runner (default: False).
        inference_mode: Enable inference mode in runner (default: True).
    """

    autocast: bool = False
    inference_mode: bool = True

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
        self.autocast = False
        self.inference_mode = True


@dataclasses.dataclass
class TorchTensorRTConfig(CustomConfigForTensorRT):
    """Torch custom config used for TensorRT TorchScript conversion."""

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
class OnnxConfig(CustomConfigForFormat):
    """ONNX custom config used for ONNX export and conversion.

    Args:
        opset: ONNX opset used for conversion.
        dynamo_export: Enable additional dynamo export.
        dynamic_axes: Dynamic axes for ONNX conversion.
        onnx_extended_conversion: Enables additional conversions from TorchScript to ONNX.
        graph_surgeon_optimization: Enables polygraphy graph surgeon optimization: fold_constants, infer_shapes, toposort, cleanup.
        export_device: Device used for ONNX export.
    """

    opset: Optional[int] = DEFAULT_ONNX_OPSET
    dynamo_export: Optional[bool] = False
    dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None
    onnx_extended_conversion: bool = False
    graph_surgeon_optimization: bool = True
    export_device: Optional[str] = None

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
        """Update parameters to defaults."""
        super().defaults()
        self.opset = DEFAULT_ONNX_OPSET
        self.dynamo_export = False
        self.dynamic_axes = None
        self.extended_conversion = False
        self.graph_surgeon_optimization = True
        self.export_device = None


@dataclasses.dataclass
class TensorRTConfig(CustomConfigForTensorRT):
    """TensorRT custom config used for TensorRT conversion.

    Args:
        optimization_level: Optimization level for TensorRT conversion. Allowed values are fom 0 to 5. Where default is
                            3 based on TensorRT API documentation.
        compatibility_level: Compatibility level for TensorRT conversion.
        onnx_parser_flags: List of TensorRT OnnxParserFlags used for conversion.
    """

    optimization_level: Optional[int] = None
    compatibility_level: Optional[TensorRTCompatibilityLevel] = None
    onnx_parser_flags: Optional[List[int]] = None

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
