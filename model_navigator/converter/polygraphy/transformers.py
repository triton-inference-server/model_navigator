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
import typing
from distutils.version import LooseVersion
from pathlib import Path
from typing import Dict, Optional

import polygraphy
import sh

from model_navigator.converter.config import DatasetProfileConfig, TensorRTPrecision
from model_navigator.converter.utils import execute_sh_command, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException, ModelNavigatorConverterException
from model_navigator.model import Format, Model
from model_navigator.tensor import TensorSpec

LOGGER = logging.getLogger("polygraphy.transformers")
DEFAULT_MAX_WORKSPACE_SIZE = 4 * 2 ** 30  # 4GB

POLYGRAPHY_VERSION = LooseVersion(polygraphy.__version__)
DEFAULT_TOLERANCES = {"rtol": 1e-5, "atol": 1e-5}


class LegacyFormatSeparator:
    name_shape_sep = ","
    shape_sep = "x"
    shape_prefix = ""
    shape_suffix = ""


class UpdatedFormatSeparators:
    name_shape_sep = ":"
    shape_sep = ","
    shape_prefix = "["
    shape_suffix = "]"


def _serialize_shapes(shapes):
    new_format = POLYGRAPHY_VERSION > LooseVersion("0.24.2")  # not sure about this version
    sep = UpdatedFormatSeparators if new_format else LegacyFormatSeparator
    return [
        f"{name}{sep.name_shape_sep}{sep.shape_prefix}{sep.shape_sep.join(map(str, shape))}{sep.shape_suffix}"
        for name, shape in shapes.items()
    ]


def _serialize_tolerance(name, value):
    sep = ":" if POLYGRAPHY_VERSION > LooseVersion("0.24.2") else ","
    return f"{name}{sep}{value}" if name else str(value)


def _validate_keys(io_type, expected_io_names, io_names):
    missing_keys = sorted(set(expected_io_names) - set(io_names))
    unknown_keys = sorted(set(io_names) - set(expected_io_names))
    if missing_keys:
        missing_keys = ", ".join(missing_keys)
        raise ModelNavigatorConverterException(f"{io_type} is defined but there are missing keys: {missing_keys}")

    if unknown_keys:
        unknown_keys = ", ".join(unknown_keys)
        expected_io_names = ", ".join(expected_io_names)
        raise ModelNavigatorConverterException(
            f"{io_type} is defined but there are unknown keys: {unknown_keys}; possible keys: {expected_io_names}"
        )


class ProfilesAdapter:
    def __init__(self, profiles: DatasetProfileConfig):
        self._profiles = profiles

    def validate_for_model(self, model: Model):
        # Polygraphy assign profile opt values from min values and max values from opt values if opt or max are missing
        # Polygraphy checks if set of names of min/opt/max profiles equals

        if not self.is_defined() and model.signature.has_input_dynamic_axes():
            input_signature_str = ", ".join([str(spec) for spec in model.signature.inputs.values()])
            raise ModelNavigatorConverterException(
                f"Missing profile definition for model with inputs containing dynamic axes. "
                f"Input signature: {input_signature_str}. "
                "Use [min|opt|max]_shapes for optimization profile definition."
            )

        inputs = model.signature.inputs or {}

        for type_name, profile_item in [
            ("min_shapes", self._profiles.min_shapes),
            ("opt_shapes", self._profiles.opt_shapes),
            ("max_shapes", self._profiles.max_shapes),
        ]:
            profile_item = profile_item or {}

            # Check if batch_size for each input are the same
            batch_sizes = sorted({shape[0] for name, shape in profile_item.items()})
            if len(batch_sizes) > 1:
                raise ModelNavigatorConverterException(
                    f"Inconsistent batch sizes in optimisation profile {type_name} {', '.join(batch_sizes)}"
                )

            # check if profile is defined at least for dynamic axes
            missing_names = [name for name, spec in inputs.items() if spec.is_dynamic() and name not in profile_item]
            if missing_names:
                raise ModelNavigatorConverterException(
                    f"There is missing shape definition for dynamic inputs: {', '.join(missing_names)} in {type_name}"
                )

    @property
    def runtime_inputs(self) -> Optional[typing.Dict[str, typing.Tuple]]:
        runtime_inputs = None
        if self._profiles:
            for type_name, profile_item in [
                ("min_shapes", self._profiles.min_shapes),
                ("opt_shapes", self._profiles.opt_shapes),
                ("max_shapes", self._profiles.max_shapes),
            ]:
                if profile_item:
                    runtime_inputs = profile_item
                    LOGGER.debug(f"Use {type_name}: {runtime_inputs} as runtime shape")
                    break
        return runtime_inputs

    @property
    def profile_flags(self) -> typing.List[str]:
        profiling_flags = []

        if self._profiles:
            for flag_name, profile_item in [
                ("--trt-min-shapes", self._profiles.min_shapes),
                ("--trt-opt-shapes", self._profiles.opt_shapes),
                ("--trt-max-shapes", self._profiles.max_shapes),
            ]:
                if profile_item:
                    profiling_flags.extend([flag_name] + _serialize_shapes(profile_item))

        return profiling_flags

    @property
    def input_flags(self) -> typing.List[str]:
        inputs_flags = []
        runtime_inputs = self.runtime_inputs
        if runtime_inputs is not None:
            # this is only for runner
            inputs_flags = ["--inputs"] + _serialize_shapes(runtime_inputs)
        return inputs_flags

    def is_defined(self):
        return self._profiles and any([self._profiles.min_shapes, self._profiles.opt_shapes, self._profiles.max_shapes])

    @classmethod
    def from_model_input(cls, model: Model, max_batch_size: int):
        def _get_profile_item(batch_size):
            return {name: (batch_size,) + spec.shape[1:] for name, spec in model.signature.inputs.items()}

        if model.signature.inputs:
            profile = DatasetProfileConfig(
                min_shapes=_get_profile_item(1),
                opt_shapes=_get_profile_item(max_batch_size),
                max_shapes=_get_profile_item(max_batch_size),
            )
        else:
            profile = None

        return cls(profile)


def _validate_names_in_comparator_config(
    outputs: Dict[str, TensorSpec], rtol: Dict[str, float], atol: Dict[str, float]
):
    def _validate_tolerance(name, outputs_: Dict[str, TensorSpec], tolerance_: Dict[str, float]):
        output_names = list(outputs_)
        tolerance_names = list(tolerance_)
        ALL_OTHER_OUTPUTS = ""
        if ALL_OTHER_OUTPUTS in tolerance_names:
            missing_names = [name for name in output_names if name not in tolerance_names]
            tolerance_names += missing_names
            tolerance_names.remove(ALL_OTHER_OUTPUTS)

        _validate_keys(name, expected_io_names=output_names, io_names=tolerance_names)

    _validate_tolerance("rtol", outputs, rtol)
    _validate_tolerance("atol", outputs, atol)


def onnx2trt(
    *,
    input_path: Path,
    output_path: Path,
    log_path: Path,
    precision: TensorRTPrecision,
    max_batch_size: Optional[int] = None,
    max_workspace_size: Optional[int] = None,
    profiles: Optional[DatasetProfileConfig] = None,
    rtol: Optional[Dict[str, float]] = None,
    atol: Optional[Dict[str, float]] = None,
    verbose: bool = False,
):
    from sh import polygraphy  # noqa

    LOGGER.info("Polygraphy onnx2trt started.")

    trt_precision_flags = {
        TensorRTPrecision.FP32: "--tf32",
        TensorRTPrecision.TF32: "--tf32",
        TensorRTPrecision.FP16: "--fp16",
    }[precision]

    LOGGER.warning("This conversion should be done on target GPU platform")

    model = Model("model_to_convert", input_path)
    profiles_adapter = ProfilesAdapter(profiles)
    if not profiles_adapter.is_defined() and not model.signature.has_input_dynamic_axes():
        if max_batch_size is None:
            raise ModelNavigatorConverterException(
                "Missing profile definition. Use [min|opt|max]_shapes for optimization profile definition or "
                "max_batch_size to create default profile."
            )
        profiles_adapter = ProfilesAdapter.from_model_input(model, max_batch_size)
    else:
        profiles_adapter.validate_for_model(model)

    _validate_names_in_comparator_config(model.signature.outputs, rtol, atol)

    # TODO: obtain free memory on gpu
    if max_workspace_size is None:
        max_workspace_size = DEFAULT_MAX_WORKSPACE_SIZE
        LOGGER.warning(f"--max-workspace-size config parameter is missing thus using {DEFAULT_MAX_WORKSPACE_SIZE}")

    tolerance_flags = []

    def _add_tolerance_params(params_name, tolerance_params):
        if tolerance_params:
            tolerance_params.setdefault("", DEFAULT_TOLERANCES[params_name])
            params = [_serialize_tolerance(name, value) for name, value in tolerance_params.items()]
            tolerance_flags.extend([f"--{params_name}"] + params)

    _add_tolerance_params("rtol", rtol or {})
    _add_tolerance_params("atol", atol or {})

    args = [
        "--onnxrt",
        "--trt",
        input_path.as_posix(),
        "--model-type",
        "onnx",
        *profiles_adapter.input_flags,
        "--onnx-outputs",
        *list(model.signature.outputs),
        "--shape-inference",
        trt_precision_flags,
        *profiles_adapter.profile_flags,
        *tolerance_flags,
        "--workspace",
        max_workspace_size,
        "--save-engine",
        output_path,
    ]
    if verbose:
        args += ["-v"]

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.ONNX, Format.TENSORRT)
            execute_sh_command(polygraphy.run.bake(*args), log_file=log_file, verbose=verbose)
        LOGGER.info("Polygraphy onnx2trt succeed.")
    except sh.ErrorReturnCode as e:
        LOGGER.warning(f"Polygraphy onnx2trt conversion failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=e.stdout.decode("utf-8"), log_path=log_path)
